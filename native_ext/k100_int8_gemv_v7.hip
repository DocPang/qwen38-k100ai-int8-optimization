#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

namespace {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource { i32x4 content; struct { uint64_t address; uint32_t range; uint32_t config; } fields; };
__device__ __forceinline__ i32x4 make_buffer_resource(const void* p,uint32_t bytes){
  WaveBufferResource r{}; uint64_t a=(uint64_t)p;
  r.content[0]=__builtin_amdgcn_readfirstlane((uint32_t)a);
  r.content[1]=__builtin_amdgcn_readfirstlane((uint32_t)(a>>32));
  r.content[2]=__builtin_amdgcn_readfirstlane(bytes); r.content[3]=0x00020000; return r.content;
}
__device__ __forceinline__ i32x4 buf_load_x4(i32x4 r,uint32_t off){
  i32x4 v; asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t":"=v"(v):"v"(off),"s"(r):"memory"); return v;
}
__device__ __forceinline__ void wait_v0(){asm volatile("s_waitcnt vmcnt(0)":::"memory");}
__device__ __forceinline__ void wait_v2(){asm volatile("s_waitcnt vmcnt(2)":::"memory");}
__device__ __forceinline__ void wait_v4(){asm volatile("s_waitcnt vmcnt(4)":::"memory");}
__device__ __forceinline__ void wait_v6(){asm volatile("s_waitcnt vmcnt(6)":::"memory");}
__device__ __forceinline__ int dot_all(i32x4 x,i32x4 w,int a){
  a=__builtin_amdgcn_sdot4((int)x[0],(int)w[0],a,false);
  a=__builtin_amdgcn_sdot4((int)x[1],(int)w[1],a,false);
  a=__builtin_amdgcn_sdot4((int)x[2],(int)w[2],a,false);
  a=__builtin_amdgcn_sdot4((int)x[3],(int)w[3],a,false);
  return a;
}

template<int WAVES,int PIPE>
__global__ __launch_bounds__(WAVES*64) void gemv_v7_k6144(const int8_t* __restrict__ x,const int8_t* __restrict__ w,
 const float* __restrict__ xs,const float* __restrict__ ws,hip_bfloat16* __restrict__ out,int n_out){
  const int lane=threadIdx.x&63,wave=threadIdx.x>>6,n=blockIdx.x*WAVES+wave; if(n>=n_out)return;
  constexpr int K_DIM=6144;
  const i32x4 xr=make_buffer_resource(x,K_DIM);
  const int8_t* wrp=w+(int64_t)n*K_DIM; const i32x4 wr=make_buffer_resource(wrp,K_DIM);
  auto off=[&](int s)->uint32_t{return uint32_t((lane+s*64)*16);};
  int acc0=0,acc1=0;
  if constexpr(PIPE==2){
    i32x4 x0=buf_load_x4(xr,off(0)), w0=buf_load_x4(wr,off(0));
    i32x4 x1=buf_load_x4(xr,off(1)), w1=buf_load_x4(wr,off(1));
    i32x4 x2=buf_load_x4(xr,off(2)), w2=buf_load_x4(wr,off(2));
    wait_v4(); acc0=dot_all(x0,w0,acc0);
    x0=buf_load_x4(xr,off(3)); w0=buf_load_x4(wr,off(3)); wait_v4(); acc1=dot_all(x1,w1,acc1);
    x1=buf_load_x4(xr,off(4)); w1=buf_load_x4(wr,off(4)); wait_v4(); acc0=dot_all(x2,w2,acc0);
    x2=buf_load_x4(xr,off(5)); w2=buf_load_x4(wr,off(5)); wait_v4(); acc1=dot_all(x0,w0,acc1);
    wait_v2(); acc0=dot_all(x1,w1,acc0);
    wait_v0(); acc1=dot_all(x2,w2,acc1);
  } else {
    static_assert(PIPE==3);
    i32x4 x0=buf_load_x4(xr,off(0)), w0=buf_load_x4(wr,off(0));
    i32x4 x1=buf_load_x4(xr,off(1)), w1=buf_load_x4(wr,off(1));
    i32x4 x2=buf_load_x4(xr,off(2)), w2=buf_load_x4(wr,off(2));
    i32x4 x3=buf_load_x4(xr,off(3)), w3=buf_load_x4(wr,off(3));
    wait_v6(); acc0=dot_all(x0,w0,acc0);
    x0=buf_load_x4(xr,off(4)); w0=buf_load_x4(wr,off(4)); wait_v6(); acc1=dot_all(x1,w1,acc1);
    x1=buf_load_x4(xr,off(5)); w1=buf_load_x4(wr,off(5)); wait_v6(); acc0=dot_all(x2,w2,acc0);
    wait_v4(); acc1=dot_all(x3,w3,acc1);
    wait_v2(); acc0=dot_all(x0,w0,acc0);
    wait_v0(); acc1=dot_all(x1,w1,acc1);
  }
  int acc=acc0+acc1;
#pragma unroll
  for(int sh=32;sh>0;sh>>=1) acc+=__shfl_down(acc,sh,64);
  if(lane==0) out[n]=hip_bfloat16((float)acc*xs[0]*ws[n]);
}

template<int W,int P> void launch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s){
  dim3 b(W*64),g((n+W-1)/W); hipLaunchKernelGGL((gemv_v7_k6144<W,P>),g,b,0,s,x,w,xs,ws,out,n);
}
void validate(const torch::Tensor&x,const torch::Tensor&w,const torch::Tensor&xs,const torch::Tensor&ws){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==torch::kInt8&&w.scalar_type()==torch::kInt8,"int8 cuda");
  TORCH_CHECK(x.dim()==2&&x.size(0)==1&&x.size(1)==6144&&x.is_contiguous()&&w.is_contiguous()&&w.size(1)==6144,"K6144 contiguous");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32,"scale f32");
}
}

torch::Tensor gemv(torch::Tensor x,torch::Tensor w,torch::Tensor xs,torch::Tensor ws,int64_t waves,int64_t pipe){
 validate(x,w,xs,ws); auto o=torch::empty({1,w.size(0)},x.options().dtype(torch::kBFloat16)); int n=w.size(0); auto*op=(hip_bfloat16*)o.data_ptr<at::BFloat16>(); hipStream_t s=c10::hip::getCurrentHIPStream();
#define L(W,P) launch<W,P>(x.data_ptr<int8_t>(),w.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),op,n,s)
#define PSEL(W) do{if(pipe==2)L(W,2);else if(pipe==3)L(W,3);else TORCH_CHECK(false,"pipe 2/3");}while(0)
 if(waves==1)PSEL(1);else if(waves==2)PSEL(2);else if(waves==4)PSEL(4);else if(waves==8)PSEL(8);else TORCH_CHECK(false,"waves");
#undef PSEL
#undef L
 return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemv",&gemv,"K100AI K6144 deeper-buffer-pipeline GEMV v7",py::arg("x"),py::arg("w"),py::arg("xs"),py::arg("ws"),py::arg("waves")=2,py::arg("pipe")=2);}
