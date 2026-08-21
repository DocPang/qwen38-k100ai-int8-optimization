#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

// U008: generalize the accepted K6144 v7 four-pair buffer pipeline to all
// production M=1 INT8 GEMV K dimensions. This keeps the exact int32 sdot4 math
// and BF16 epilogue; only the memory pipeline is changed versus generic_v1.
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

template<int K_DIM,int WAVES,int PIPE>
__global__ __launch_bounds__(WAVES*64) void gemv_generic_v2(
 const int8_t* __restrict__ x,const int8_t* __restrict__ w,
 const float* __restrict__ xs,const float* __restrict__ ws,
 hip_bfloat16* __restrict__ out,int n_out){
  static_assert(K_DIM % 1024 == 0);
  constexpr int NSEGS=K_DIM/1024;
  static_assert(NSEGS>=4);
  const int lane=threadIdx.x&63,wave=threadIdx.x>>6,n=blockIdx.x*WAVES+wave;
  if(n>=n_out)return;
  const i32x4 xr=make_buffer_resource(x,K_DIM);
  const int8_t* wrp=w+(int64_t)n*K_DIM;
  const i32x4 wr=make_buffer_resource(wrp,K_DIM);
  auto off=[&](int s)->uint32_t{return uint32_t((lane+s*64)*16);};
  int acc0=0,acc1=0;
  if constexpr(PIPE==2){
    // Keep generic_v1's two-pair rolling pipeline as a control.
    i32x4 x0=buf_load_x4(xr,off(0)), w0=buf_load_x4(wr,off(0));
    i32x4 x1=buf_load_x4(xr,off(1)), w1=buf_load_x4(wr,off(1));
#pragma unroll
    for(int s=2;s<NSEGS;s++){
      wait_v2();
      if((s-2)&1) acc1=dot_all(x0,w0,acc1); else acc0=dot_all(x0,w0,acc0);
      x0=x1; w0=w1;
      x1=buf_load_x4(xr,off(s)); w1=buf_load_x4(wr,off(s));
    }
    wait_v2();
    if((NSEGS-2)&1) acc1=dot_all(x0,w0,acc1); else acc0=dot_all(x0,w0,acc0);
    wait_v0();
    if((NSEGS-1)&1) acc1=dot_all(x1,w1,acc1); else acc0=dot_all(x1,w1,acc0);
  } else {
    static_assert(PIPE==3);
    // Four segment pairs (8 vector memory ops) in flight. wait_v6 guarantees
    // the oldest x/w pair is complete while leaving the following three pairs
    // outstanding. This is the accepted K6144 v7 pattern generalized to any
    // compile-time NSEGS in the production set.
    i32x4 xv[4],wv[4];
#pragma unroll
    for(int s=0;s<4;s++){
      xv[s]=buf_load_x4(xr,off(s));
      wv[s]=buf_load_x4(wr,off(s));
    }
#pragma unroll
    for(int s=4;s<NSEGS;s++){
      wait_v6();
      constexpr bool impossible=false; (void)impossible;
      const int old=s-4;
      const int slot=old&3;
      if(old&1) acc1=dot_all(xv[slot],wv[slot],acc1);
      else acc0=dot_all(xv[slot],wv[slot],acc0);
      xv[slot]=buf_load_x4(xr,off(s));
      wv[slot]=buf_load_x4(wr,off(s));
    }
    // Exactly four segment pairs remain in flight, oldest first.
#pragma unroll
    for(int j=0;j<4;j++){
      if(j==0) wait_v6();
      else if(j==1) wait_v4();
      else if(j==2) wait_v2();
      else wait_v0();
      const int seg=NSEGS-4+j;
      const int slot=seg&3;
      if(seg&1) acc1=dot_all(xv[slot],wv[slot],acc1);
      else acc0=dot_all(xv[slot],wv[slot],acc0);
    }
  }
  int acc=acc0+acc1;
#pragma unroll
  for(int sh=32;sh>0;sh>>=1) acc+=__shfl_down(acc,sh,64);
  if(lane==0) out[n]=hip_bfloat16((float)acc*xs[0]*ws[n]);
}

template<int K,int W,int P> void launch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s){
  dim3 b(W*64),g((n+W-1)/W); hipLaunchKernelGGL((gemv_generic_v2<K,W,P>),g,b,0,s,x,w,xs,ws,out,n);
}
template<int K> void dispatch_wp(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s,int64_t waves,int64_t pipe){
#define LP(W,P) launch<K,W,P>(x,w,xs,ws,out,n,s)
#define PSEL(W) do{if(pipe==2)LP(W,2);else if(pipe==3)LP(W,3);else TORCH_CHECK(false,"pipe 2/3");}while(0)
  if(waves==1)PSEL(1); else if(waves==2)PSEL(2); else if(waves==4)PSEL(4); else if(waves==8)PSEL(8); else TORCH_CHECK(false,"waves 1/2/4/8");
#undef PSEL
#undef LP
}
void validate(const torch::Tensor&x,const torch::Tensor&w,const torch::Tensor&xs,const torch::Tensor&ws){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==torch::kInt8&&w.scalar_type()==torch::kInt8,"int8 cuda");
  TORCH_CHECK(x.dim()==2&&x.size(0)==1&&x.is_contiguous()&&w.dim()==2&&w.is_contiguous()&&w.size(1)==x.size(1),"M1 contiguous NxK");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32,"scale f32");
  TORCH_CHECK(xs.numel()>=1&&ws.numel()>=w.size(0),"scale shapes");
  auto k=x.size(1); TORCH_CHECK(k==5120||k==6144||k==10240||k==17408,"supported K");
}
}

torch::Tensor gemv(torch::Tensor x,torch::Tensor w,torch::Tensor xs,torch::Tensor ws,int64_t waves,int64_t pipe){
  validate(x,w,xs,ws);
  auto o=torch::empty({1,w.size(0)},x.options().dtype(torch::kBFloat16));
  int n=w.size(0); auto*op=(hip_bfloat16*)o.data_ptr<at::BFloat16>();
  hipStream_t s=c10::hip::getCurrentHIPStream();
  auto k=x.size(1); const auto*xptr=x.data_ptr<int8_t>(); const auto*wptr=w.data_ptr<int8_t>(); const auto*xsp=xs.data_ptr<float>(); const auto*wsp=ws.data_ptr<float>();
  if(k==5120) dispatch_wp<5120>(xptr,wptr,xsp,wsp,op,n,s,waves,pipe);
  else if(k==6144) dispatch_wp<6144>(xptr,wptr,xsp,wsp,op,n,s,waves,pipe);
  else if(k==10240) dispatch_wp<10240>(xptr,wptr,xsp,wsp,op,n,s,waves,pipe);
  else if(k==17408) dispatch_wp<17408>(xptr,wptr,xsp,wsp,op,n,s,waves,pipe);
  return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemv",&gemv,"K100AI generic M1 INT8 GEMV v2 four-pair pipeline",py::arg("x"),py::arg("w"),py::arg("xs"),py::arg("ws"),py::arg("waves")=2,py::arg("pipe")=3);}
