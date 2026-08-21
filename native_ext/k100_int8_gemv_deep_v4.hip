#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

// U015: accepted U008 row1 GEMV with deeper memory pipeline only.
// Each segment issues x+w (2 VMEM ops). DEPTH D keeps D segment pairs in flight;
// oldest pair is safe when vmcnt <= 2*(D-1). Scan D={4,6,8} for long-K down.
namespace {
using i32x4=int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource{i32x4 content;struct{uint64_t address;uint32_t range;uint32_t config;}fields;};
__device__ __forceinline__ i32x4 make_res(const void*p,uint32_t bytes){WaveBufferResource r{};uint64_t a=(uint64_t)p;r.content[0]=__builtin_amdgcn_readfirstlane((uint32_t)a);r.content[1]=__builtin_amdgcn_readfirstlane((uint32_t)(a>>32));r.content[2]=__builtin_amdgcn_readfirstlane(bytes);r.content[3]=0x00020000;return r.content;}
__device__ __forceinline__ i32x4 ld4(i32x4 r,uint32_t off){i32x4 v;asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t":"=v"(v):"v"(off),"s"(r):"memory");return v;}
#define WV(N) __device__ __forceinline__ void wait##N(){asm volatile("s_waitcnt vmcnt(" #N ")":::"memory");}
WV(0) WV(2) WV(4) WV(6) WV(8) WV(10) WV(12) WV(14)
#undef WV
__device__ __forceinline__ int dot_all(i32x4 x,i32x4 w,int a){a=__builtin_amdgcn_sdot4((int)x[0],(int)w[0],a,false);a=__builtin_amdgcn_sdot4((int)x[1],(int)w[1],a,false);a=__builtin_amdgcn_sdot4((int)x[2],(int)w[2],a,false);a=__builtin_amdgcn_sdot4((int)x[3],(int)w[3],a,false);return a;}
template<int D> __device__ __forceinline__ void wait_old(){if constexpr(D==4)wait6();else if constexpr(D==6)wait10();else{static_assert(D==8);wait14();}}
template<int D,int J> __device__ __forceinline__ void wait_tail(){constexpr int v=2*(D-1-J);if constexpr(v==14)wait14();else if constexpr(v==12)wait12();else if constexpr(v==10)wait10();else if constexpr(v==8)wait8();else if constexpr(v==6)wait6();else if constexpr(v==4)wait4();else if constexpr(v==2)wait2();else wait0();}

template<int K_DIM,int WAVES,int DEPTH>
__global__ __launch_bounds__(WAVES*64) void gemv_deep_v4(
 const int8_t* __restrict__ x,const int8_t* __restrict__ w,
 const float* __restrict__ xs,const float* __restrict__ ws,
 hip_bfloat16* __restrict__ out,int nout){
 static_assert(K_DIM%1024==0);constexpr int NS=K_DIM/1024;static_assert(NS>=DEPTH);
 int lane=threadIdx.x&63,wave=threadIdx.x>>6,n=blockIdx.x*WAVES+wave;if(n>=nout)return;
 i32x4 xr=make_res(x,K_DIM),wr=make_res(w+(int64_t)n*K_DIM,K_DIM);auto off=[&](int s)->uint32_t{return uint32_t((lane+s*64)*16);};
 i32x4 xv[DEPTH],wv[DEPTH];
#pragma unroll
 for(int s=0;s<DEPTH;s++){xv[s]=ld4(xr,off(s));wv[s]=ld4(wr,off(s));}
 int a0=0,a1=0;
#pragma unroll
 for(int s=DEPTH;s<NS;s++){wait_old<DEPTH>();int old=s-DEPTH,slot=old%DEPTH;if(old&1)a1=dot_all(xv[slot],wv[slot],a1);else a0=dot_all(xv[slot],wv[slot],a0);xv[slot]=ld4(xr,off(s));wv[slot]=ld4(wr,off(s));}
 if constexpr(DEPTH==4){
#pragma unroll
  for(int j=0;j<4;j++){if(j==0)wait_tail<4,0>();else if(j==1)wait_tail<4,1>();else if(j==2)wait_tail<4,2>();else wait_tail<4,3>();int seg=NS-4+j,slot=seg%4;if(seg&1)a1=dot_all(xv[slot],wv[slot],a1);else a0=dot_all(xv[slot],wv[slot],a0);}
 }else if constexpr(DEPTH==6){
#pragma unroll
  for(int j=0;j<6;j++){if(j==0)wait_tail<6,0>();else if(j==1)wait_tail<6,1>();else if(j==2)wait_tail<6,2>();else if(j==3)wait_tail<6,3>();else if(j==4)wait_tail<6,4>();else wait_tail<6,5>();int seg=NS-6+j,slot=seg%6;if(seg&1)a1=dot_all(xv[slot],wv[slot],a1);else a0=dot_all(xv[slot],wv[slot],a0);}
 }else{
#pragma unroll
  for(int j=0;j<8;j++){if(j==0)wait_tail<8,0>();else if(j==1)wait_tail<8,1>();else if(j==2)wait_tail<8,2>();else if(j==3)wait_tail<8,3>();else if(j==4)wait_tail<8,4>();else if(j==5)wait_tail<8,5>();else if(j==6)wait_tail<8,6>();else wait_tail<8,7>();int seg=NS-8+j,slot=seg%8;if(seg&1)a1=dot_all(xv[slot],wv[slot],a1);else a0=dot_all(xv[slot],wv[slot],a0);}
 }
 int acc=a0+a1;
#pragma unroll
 for(int sh=32;sh>0;sh>>=1)acc+=__shfl_down(acc,sh,64);
 if(lane==0)out[n]=hip_bfloat16((float)acc*xs[0]*ws[n]);
}

template<int W,int D> void launch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s){
 dim3 b(W*64),g((n+W-1)/W); hipLaunchKernelGGL((gemv_deep_v4<17408,W,D>),g,b,0,s,x,w,xs,ws,out,n);
}
void dispatch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s,int64_t waves,int64_t depth){
#define L(W,D) launch<W,D>(x,w,xs,ws,out,n,s)
#define DS(W) do{if(depth==4)L(W,4);else if(depth==6)L(W,6);else if(depth==8)L(W,8);else TORCH_CHECK(false,"depth 4/6/8");}while(0)
 if(waves==1)DS(1);else if(waves==2)DS(2);else if(waves==4)DS(4);else TORCH_CHECK(false,"waves 1/2/4");
#undef DS
#undef L
}
void validate(const torch::Tensor&x,const torch::Tensor&w,const torch::Tensor&xs,const torch::Tensor&ws){
 TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==torch::kInt8&&w.scalar_type()==torch::kInt8,"int8 cuda");
 TORCH_CHECK(x.dim()==2&&x.size(0)==1&&x.size(1)==17408&&x.is_contiguous()&&w.dim()==2&&w.is_contiguous()&&w.size(1)==17408,"M1 K17408 contiguous NxK");
 TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32&&xs.numel()>=1&&ws.numel()>=w.size(0),"f32 scales");
}
}
torch::Tensor gemv(torch::Tensor x,torch::Tensor w,torch::Tensor xs,torch::Tensor ws,int64_t waves,int64_t depth){
 validate(x,w,xs,ws); auto o=torch::empty({1,w.size(0)},x.options().dtype(torch::kBFloat16)); auto s=c10::hip::getCurrentHIPStream(); int n=w.size(0);
 auto*xp=x.data_ptr<int8_t>();auto*wp=w.data_ptr<int8_t>();auto*xsp=xs.data_ptr<float>();auto*wsp=ws.data_ptr<float>();auto*op=(hip_bfloat16*)o.data_ptr<at::BFloat16>();
 dispatch(xp,wp,xsp,wsp,op,n,s,waves,depth); return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemv",&gemv,"K100AI M1 INT8 GEMV deep v4",py::arg("x"),py::arg("w"),py::arg("xs"),py::arg("ws"),py::arg("waves")=1,py::arg("depth")=4);}
