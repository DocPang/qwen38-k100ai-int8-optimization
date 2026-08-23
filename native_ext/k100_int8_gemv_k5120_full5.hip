#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

// U017: K5120/M1 short-K specialization.
// K=5120 is exactly five 1024-byte wave segments. generic-v2(pipe=3) keeps
// four x/w segment pairs in flight, consumes segment0, then issues segment4.
// Here all five x/w pairs are issued up-front and then drained with
// vmcnt 8/6/4/2/0. Arithmetic (even/odd int32 chains + final add) stays
// identical to generic-v2; only the short-K memory schedule changes.
namespace {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource {
  i32x4 content;
  struct { uint64_t address; uint32_t range; uint32_t config; } fields;
};

__device__ __forceinline__ i32x4 make_res(const void* p, uint32_t bytes) {
  WaveBufferResource r{};
  uint64_t a = (uint64_t)p;
  r.content[0] = __builtin_amdgcn_readfirstlane((uint32_t)a);
  r.content[1] = __builtin_amdgcn_readfirstlane((uint32_t)(a >> 32));
  r.content[2] = __builtin_amdgcn_readfirstlane(bytes);
  r.content[3] = 0x00020000;
  return r.content;
}

__device__ __forceinline__ i32x4 ld4(i32x4 r, uint32_t off) {
  i32x4 v;
  asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t"
               : "=v"(v) : "v"(off), "s"(r) : "memory");
  return v;
}
#define WV(N) __device__ __forceinline__ void wait##N(){asm volatile("s_waitcnt vmcnt(" #N ")":::"memory");}
WV(0) WV(2) WV(4) WV(6) WV(8)
#undef WV

__device__ __forceinline__ int dot_all(i32x4 x, i32x4 w, int a) {
  a = __builtin_amdgcn_sdot4((int)x[0], (int)w[0], a, false);
  a = __builtin_amdgcn_sdot4((int)x[1], (int)w[1], a, false);
  a = __builtin_amdgcn_sdot4((int)x[2], (int)w[2], a, false);
  a = __builtin_amdgcn_sdot4((int)x[3], (int)w[3], a, false);
  return a;
}

template<int WAVES>
__global__ __launch_bounds__(WAVES * 64) void gemv_k5120_full5(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ xs,
    const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out,
    int nout) {
  constexpr int K = 5120;
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= nout) return;

  const i32x4 xr = make_res(x, K);
  const i32x4 wr = make_res(w + (int64_t)n * K, K);
  auto off = [&](int s) -> uint32_t { return uint32_t((lane + s * 64) * 16); };

  i32x4 xv[5], wv[5];
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    xv[s] = ld4(xr, off(s));
    wv[s] = ld4(wr, off(s));
  }

  int a0 = 0, a1 = 0;
  wait8(); a0 = dot_all(xv[0], wv[0], a0);
  wait6(); a1 = dot_all(xv[1], wv[1], a1);
  wait4(); a0 = dot_all(xv[2], wv[2], a0);
  wait2(); a1 = dot_all(xv[3], wv[3], a1);
  wait0(); a0 = dot_all(xv[4], wv[4], a0);

  int acc = a0 + a1;
#pragma unroll
  for (int sh = 32; sh > 0; sh >>= 1) acc += __shfl_down(acc, sh, 64);
  if (lane == 0) out[n] = hip_bfloat16((float)acc * xs[0] * ws[n]);
}

template<int W>
void launch(const int8_t* x, const int8_t* w, const float* xs, const float* ws,
            hip_bfloat16* out, int n, hipStream_t s) {
  dim3 b(W * 64), g((n + W - 1) / W);
  hipLaunchKernelGGL((gemv_k5120_full5<W>), g, b, 0, s, x, w, xs, ws, out, n);
}

void dispatch(const int8_t* x, const int8_t* w, const float* xs, const float* ws,
              hip_bfloat16* out, int n, hipStream_t s, int64_t waves) {
  if (waves == 1) launch<1>(x,w,xs,ws,out,n,s);
  else if (waves == 2) launch<2>(x,w,xs,ws,out,n,s);
  else if (waves == 4) launch<4>(x,w,xs,ws,out,n,s);
  else if (waves == 8) launch<8>(x,w,xs,ws,out,n,s);
  else TORCH_CHECK(false, "waves must be 1/2/4/8");
}

void validate(const torch::Tensor& x, const torch::Tensor& w,
              const torch::Tensor& xs, const torch::Tensor& ws) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kInt8 && w.scalar_type() == torch::kInt8, "int8 x/w required");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == 5120 && x.is_contiguous(), "x must be contiguous [1,5120]");
  TORCH_CHECK(w.dim() == 2 && w.size(1) == 5120 && w.is_contiguous(), "w must be contiguous [N,5120]");
  TORCH_CHECK(xs.scalar_type() == torch::kFloat32 && ws.scalar_type() == torch::kFloat32, "scale f32 required");
  TORCH_CHECK(xs.numel() >= 1 && ws.numel() >= w.size(0), "scale shapes invalid");
}
} // namespace

torch::Tensor gemv(torch::Tensor x, torch::Tensor w, torch::Tensor xs,
                   torch::Tensor ws, int64_t waves) {
  validate(x,w,xs,ws);
  auto o = torch::empty({1, w.size(0)}, x.options().dtype(torch::kBFloat16));
  auto s = c10::hip::getCurrentHIPStream();
  dispatch(x.data_ptr<int8_t>(), w.data_ptr<int8_t>(), xs.data_ptr<float>(),
           ws.data_ptr<float>(), (hip_bfloat16*)o.data_ptr<at::BFloat16>(),
           (int)w.size(0), s, waves);
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv, "K100AI M1 K5120 full-five-prefetch INT8 GEMV",
        py::arg("x"), py::arg("w"), py::arg("xs"), py::arg("ws"), py::arg("waves") = 4);
}
