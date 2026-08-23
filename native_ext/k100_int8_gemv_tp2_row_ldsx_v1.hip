#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

// TP2 row-parallel M=1 INT8 GEMV for Qwen3.8-27B:
//   down:   K8704 -> N5120
//   output: K3072 -> N5120
// One block caches the rank-local activation in LDS; each wave streams one
// output row of W. Integer dot accumulation is exact and BF16 conversion is
// performed only once after the full int32 reduction.
namespace {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource { i32x4 content; struct { uint64_t address; uint32_t range; uint32_t config; } fields; };
__device__ __forceinline__ i32x4 make_res(const void* p,uint32_t bytes){
  WaveBufferResource r{}; uint64_t a=(uint64_t)p;
  r.content[0]=__builtin_amdgcn_readfirstlane((uint32_t)a);
  r.content[1]=__builtin_amdgcn_readfirstlane((uint32_t)(a>>32));
  r.content[2]=__builtin_amdgcn_readfirstlane(bytes); r.content[3]=0x00020000; return r.content;
}
__device__ __forceinline__ i32x4 ld4(i32x4 r,uint32_t off){ i32x4 v; asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t":"=v"(v):"v"(off),"s"(r):"memory"); return v; }
#define WV(N) __device__ __forceinline__ void wait##N(){asm volatile("s_waitcnt vmcnt(" #N ")":::"memory");}
WV(0) WV(1) WV(2) WV(3)
#undef WV
__device__ __forceinline__ int dot4x4(i32x4 x,i32x4 w,int a){
  a=__builtin_amdgcn_sdot4((int)x[0],(int)w[0],a,false);
  a=__builtin_amdgcn_sdot4((int)x[1],(int)w[1],a,false);
  a=__builtin_amdgcn_sdot4((int)x[2],(int)w[2],a,false);
  a=__builtin_amdgcn_sdot4((int)x[3],(int)w[3],a,false);
  return a;
}

template<int K,int WAVES>
__global__ __launch_bounds__(WAVES*64) void gemv_tp2_row_ldsx(
 const int8_t* __restrict__ x,const int8_t* __restrict__ w,
 const float* __restrict__ xs,const float* __restrict__ ws,
 hip_bfloat16* __restrict__ out,int nout){
  static_assert(K==8704 || K==3072);
  static_assert(K%16==0);
  constexpr int V=K/16;
  __shared__ __align__(16) int8_t sx[K];
  auto* sxv=reinterpret_cast<i32x4*>(sx);
  const auto* xgv=reinterpret_cast<const i32x4*>(x);
  const int tid=threadIdx.x;
#pragma unroll 1
  for(int i=tid;i<V;i+=WAVES*64) sxv[i]=xgv[i];
  __syncthreads();

  const int lane=tid&63,wave=tid>>6,n=blockIdx.x*WAVES+wave;
  if(n>=nout)return;
  const i32x4 wr=make_res(w+(int64_t)n*K,K);
  i32x4 z{};
  int a0=0,a1=0;
  if constexpr(K==8704){
    // 8704 / 16 = 544 vectors = 8 full 64-lane stripes + 32 lanes.
    i32x4 w0=ld4(wr,uint32_t((lane+0*64)*16));
    i32x4 w1=ld4(wr,uint32_t((lane+1*64)*16));
    i32x4 w2=ld4(wr,uint32_t((lane+2*64)*16));
    i32x4 w3=ld4(wr,uint32_t((lane+3*64)*16));
    i32x4 x0=sxv[lane+0*64],x1=sxv[lane+1*64],x2=sxv[lane+2*64],x3=sxv[lane+3*64];
    wait3(); a0=dot4x4(x0,w0,a0);
    wait2(); a1=dot4x4(x1,w1,a1);
    wait1(); a0=dot4x4(x2,w2,a0);
    wait0(); a1=dot4x4(x3,w3,a1);

    i32x4 w4=ld4(wr,uint32_t((lane+4*64)*16));
    i32x4 w5=ld4(wr,uint32_t((lane+5*64)*16));
    i32x4 w6=ld4(wr,uint32_t((lane+6*64)*16));
    i32x4 w7=ld4(wr,uint32_t((lane+7*64)*16));
    i32x4 x4=sxv[lane+4*64],x5=sxv[lane+5*64],x6=sxv[lane+6*64],x7=sxv[lane+7*64];
    wait3(); a0=dot4x4(x4,w4,a0);
    wait2(); a1=dot4x4(x5,w5,a1);
    wait1(); a0=dot4x4(x6,w6,a0);
    wait0(); a1=dot4x4(x7,w7,a1);

    i32x4 w8=z;
    if(lane<32) w8=ld4(wr,uint32_t((lane+8*64)*16));
    i32x4 x8=(lane<32)?sxv[lane+8*64]:z;
    wait0(); a0=dot4x4(x8,w8,a0);
  } else {
    // 3072 / 16 = 192 vectors = exactly 3 full 64-lane stripes.
    i32x4 w0=ld4(wr,uint32_t((lane+0*64)*16));
    i32x4 w1=ld4(wr,uint32_t((lane+1*64)*16));
    i32x4 w2=ld4(wr,uint32_t((lane+2*64)*16));
    i32x4 x0=sxv[lane+0*64],x1=sxv[lane+1*64],x2=sxv[lane+2*64];
    wait2(); a0=dot4x4(x0,w0,a0);
    wait1(); a1=dot4x4(x1,w1,a1);
    wait0(); a0=dot4x4(x2,w2,a0);
  }
  int acc=a0+a1;
#pragma unroll
  for(int sh=32;sh>0;sh>>=1) acc+=__shfl_down(acc,sh,64);
  if(lane==0) out[n]=hip_bfloat16((float)acc*xs[0]*ws[n]);
}

template<int K,int W> void launch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s){
  dim3 b(W*64),g((n+W-1)/W); hipLaunchKernelGGL((gemv_tp2_row_ldsx<K,W>),g,b,0,s,x,w,xs,ws,out,n);
}
template<int K> void dispatch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s,int64_t waves){
  if(waves==2)launch<K,2>(x,w,xs,ws,out,n,s); else if(waves==4)launch<K,4>(x,w,xs,ws,out,n,s); else if(waves==8)launch<K,8>(x,w,xs,ws,out,n,s); else TORCH_CHECK(false,"waves 2/4/8");
}
void validate(const torch::Tensor&x,const torch::Tensor&w,const torch::Tensor&xs,const torch::Tensor&ws){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==torch::kInt8&&w.scalar_type()==torch::kInt8,"int8 cuda required");
  TORCH_CHECK(x.dim()==2&&x.size(0)==1&&x.is_contiguous(),"x must [1,K] contiguous");
  TORCH_CHECK(w.dim()==2&&w.size(0)==5120&&w.size(1)==x.size(1)&&w.is_contiguous(),"w must [5120,K] contiguous");
  TORCH_CHECK(x.size(1)==8704||x.size(1)==3072,"K must 8704/3072");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32&&xs.numel()>=1&&ws.numel()>=5120,"scale f32 shapes");
}
}

torch::Tensor gemv(torch::Tensor x,torch::Tensor w,torch::Tensor xs,torch::Tensor ws,int64_t waves){
 validate(x,w,xs,ws); auto o=torch::empty({1,5120},x.options().dtype(torch::kBFloat16)); auto s=c10::hip::getCurrentHIPStream();
 auto*xp=x.data_ptr<int8_t>();auto*wp=w.data_ptr<int8_t>();auto*xsp=xs.data_ptr<float>();auto*wsp=ws.data_ptr<float>();auto*op=(hip_bfloat16*)o.data_ptr<at::BFloat16>();
 if(x.size(1)==8704)dispatch<8704>(xp,wp,xsp,wsp,op,5120,s,waves); else dispatch<3072>(xp,wp,xsp,wsp,op,5120,s,waves); return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemv",&gemv,"K100 TP2 row-parallel LDS-x M1 INT8 GEMV",py::arg("x"),py::arg("w"),py::arg("xs"),py::arg("ws"),py::arg("waves")=4);}
