#include "solve.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace {
__device__ __constant__ uint64_t D_FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL};
__device__ __constant__ uint64_t D_FP_P_INV = 0x89f3fffcfffcfffdULL;
__device__ __constant__ uint64_t D_FP_ONE_M[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL};
__device__ __forceinline__ bool fp_is_zero(const uint64_t* a) {
    uint64_t x=0; for(int i=0;i<6;++i) x|=a[i]; return x==0;}
__device__ __forceinline__ bool fp_eq(const uint64_t* a,const uint64_t* b) {
    for(int i=0;i<6;++i) if(a[i]!=b[i]) return false; return true;}
__device__ __forceinline__ void fp_copy(uint64_t* r,const uint64_t* a) {
    for(int i=0;i<6;++i) r[i]=a[i];}
__device__ __forceinline__ void fp_set_zero(uint64_t* r) {
    for(int i=0;i<6;++i) r[i]=0;}
__device__ __forceinline__ bool fp_ge_p(const uint64_t* a) {
    for(int i=5;i>=0;--i){if(a[i]>D_FP_P[i]) return true; if(a[i]<D_FP_P[i]) return false;} return true;}
__device__ __forceinline__ void fp_add(uint64_t* r,const uint64_t* a,const uint64_t* b) {
    uint64_t t[6]; unsigned long long carry=0;
    for(int i=0;i<6;++i){unsigned __int128 s=(unsigned __int128)a[i]+b[i]+carry; t[i]=(uint64_t)s; carry=(uint64_t)(s>>64);}
    if((carry!=0)||fp_ge_p(t)){unsigned long long borrow=0;
    for(int i=0;i<6;++i){unsigned __int128 s=(unsigned __int128)t[i]-D_FP_P[i]-borrow; r[i]=(uint64_t)s; borrow=(s>>127)?1ULL:0ULL;}
    } else {fp_copy(r,t);}}
__device__ __forceinline__ void fp_sub(uint64_t* r,const uint64_t* a,const uint64_t* b) {
    uint64_t t[6]; unsigned long long borrow=0;
    for(int i=0;i<6;++i){unsigned __int128 s=(unsigned __int128)a[i]-b[i]-borrow; t[i]=(uint64_t)s; borrow=(s>>127)?1ULL:0ULL;}
    if(borrow){unsigned long long carry=0;
    for(int i=0;i<6;++i){unsigned __int128 s=(unsigned __int128)t[i]+D_FP_P[i]+carry; r[i]=(uint64_t)s; carry=(uint64_t)(s>>64);}
    } else {fp_copy(r,t);}}
__device__ void fp_mul(uint64_t* r,const uint64_t* a,const uint64_t* b) {
    unsigned long long t[8]={0};
    for(int i=0;i<6;++i){
        unsigned long long carry=0;
        for(int j=0;j<6;++j){unsigned __int128 s=(unsigned __int128)a[j]*b[i]+t[j]+carry; t[j]=(unsigned long long)s; carry=(unsigned long long)(s>>64);}
        unsigned __int128 s6=(unsigned __int128)t[6]+carry; t[6]=(unsigned long long)s6; t[7]+=(unsigned long long)(s6>>64);
        unsigned long long m=t[0]*D_FP_P_INV; carry=0;
        for(int j=0;j<6;++j){unsigned __int128 s2=(unsigned __int128)m*D_FP_P[j]+t[j]+carry; t[j]=(unsigned long long)s2; carry=(unsigned long long)(s2>>64);}
        unsigned __int128 s3=(unsigned __int128)t[6]+carry; t[6]=(unsigned long long)s3; t[7]+=(unsigned long long)(s3>>64);
        for(int j=0;j<7;++j) t[j]=t[j+1]; t[7]=0;}
    uint64_t out[6]; for(int i=0;i<6;++i) out[i]=t[i];
    if(t[6]||fp_ge_p(out)){unsigned long long borrow=0;
    for(int i=0;i<6;++i){unsigned __int128 s=(unsigned __int128)out[i]-D_FP_P[i]-borrow; r[i]=(uint64_t)s; borrow=(s>>127)?1ULL:0ULL;}
    } else {fp_copy(r,out);}}
__device__ __forceinline__ void fp_sqr(uint64_t* r,const uint64_t* a){fp_mul(r,a,a);}
__device__ void fp_inv(uint64_t* r,const uint64_t* a) {
    const uint64_t pm2[6]={0xb9feffffffffaaa9ULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
    uint64_t result[6]; fp_copy(result,D_FP_ONE_M);
    uint64_t base[6]; fp_copy(base,a);
    for(int word=0;word<6;++word) for(int bit=0;bit<64;++bit){
        if((pm2[word]>>bit)&1ULL) fp_mul(result,result,base); fp_sqr(base,base);}
    fp_copy(r,result);}
struct JacPoint{uint64_t X[6],Y[6],Z[6];};
__device__ __forceinline__ bool jac_is_zero(const JacPoint& p){return fp_is_zero(p.Z);}
__device__ __forceinline__ void jac_set_zero(JacPoint& p){fp_set_zero(p.X);fp_set_zero(p.Y);fp_set_zero(p.Z);}
__device__ void jac_double(JacPoint& r,const JacPoint& p) {
    if(jac_is_zero(p)){jac_set_zero(r);return;}
    uint64_t A[6],B[6],C[6],D[6],E[6],F[6],tmp[6];
    fp_sqr(A,p.X);fp_sqr(B,p.Y);fp_sqr(C,B);
    fp_add(tmp,p.X,B);fp_sqr(D,tmp);fp_sub(D,D,A);fp_sub(D,D,C);fp_add(D,D,D);
    fp_add(E,A,A);fp_add(E,E,A);fp_sqr(F,E);
    uint64_t nX[6],nY[6],nZ[6];
    fp_sub(nX,F,D);fp_sub(nX,nX,D);
    fp_add(tmp,C,C);fp_add(tmp,tmp,tmp);fp_add(tmp,tmp,tmp);
    fp_sub(nY,D,nX);fp_mul(nY,E,nY);fp_sub(nY,nY,tmp);
    fp_mul(tmp,p.Y,p.Z);fp_add(nZ,tmp,tmp);
    fp_copy(r.X,nX);fp_copy(r.Y,nY);fp_copy(r.Z,nZ);}
__device__ void jac_add_affine(JacPoint& r,const JacPoint& p,const uint64_t* qx,const uint64_t* qy) {
    bool q_zero=fp_is_zero(qx)&&fp_is_zero(qy);
    if(q_zero){r=p;return;}
    if(jac_is_zero(p)){fp_copy(r.X,qx);fp_copy(r.Y,qy);fp_copy(r.Z,D_FP_ONE_M);return;}
    uint64_t Z1Z1[6],U2[6],S2[6],H[6],HH[6],I[6],J[6],rr[6],V[6],tmp[6];
    fp_sqr(Z1Z1,p.Z);fp_mul(U2,qx,Z1Z1);
    fp_mul(tmp,qy,p.Z);fp_mul(S2,tmp,Z1Z1);
    if(fp_eq(U2,p.X)){if(fp_eq(S2,p.Y)){JacPoint qj;fp_copy(qj.X,qx);fp_copy(qj.Y,qy);fp_copy(qj.Z,D_FP_ONE_M);jac_double(r,qj);return;}
        jac_set_zero(r);return;}
    fp_sub(H,U2,p.X);fp_sqr(HH,H);fp_add(I,HH,HH);fp_add(I,I,I);
    fp_mul(J,H,I);fp_sub(rr,S2,p.Y);fp_add(rr,rr,rr);fp_mul(V,p.X,I);
    uint64_t nX[6],nY[6],nZ[6];
    fp_sqr(tmp,rr);fp_sub(tmp,tmp,J);fp_sub(tmp,tmp,V);fp_sub(nX,tmp,V);
    fp_sub(tmp,V,nX);fp_mul(tmp,rr,tmp);
    uint64_t s1j[6];fp_mul(s1j,p.Y,J);fp_add(s1j,s1j,s1j);
    fp_sub(nY,tmp,s1j);
    fp_mul(nZ,p.Z,H);fp_add(nZ,nZ,nZ);
    fp_copy(r.X,nX);fp_copy(r.Y,nY);fp_copy(r.Z,nZ);}
__device__ void jac_add(JacPoint& r,const JacPoint& p,const JacPoint& q) {
    if(jac_is_zero(p)){r=q;return;}
    if(jac_is_zero(q)){r=p;return;}
    uint64_t Z1Z1[6],Z2Z2[6],U1[6],U2[6],S1[6],S2[6],H[6],I[6],J[6],rr[6],V[6],tmp[6];
    fp_sqr(Z1Z1,p.Z);fp_sqr(Z2Z2,q.Z);
    fp_mul(U1,p.X,Z2Z2);fp_mul(U2,q.X,Z1Z1);
    fp_mul(tmp,p.Y,q.Z);fp_mul(S1,tmp,Z2Z2);
    fp_mul(tmp,q.Y,p.Z);fp_mul(S2,tmp,Z1Z1);
    if(fp_eq(U1,U2)){if(fp_eq(S1,S2)){jac_double(r,p);return;} jac_set_zero(r);return;}
    fp_sub(H,U2,U1);fp_add(tmp,H,H);fp_sqr(I,tmp);
    fp_mul(J,H,I);fp_sub(rr,S2,S1);fp_add(rr,rr,rr);fp_mul(V,U1,I);
    uint64_t nX[6],nY[6],nZ[6];
    fp_sqr(tmp,rr);fp_sub(tmp,tmp,J);fp_sub(tmp,tmp,V);fp_sub(nX,tmp,V);
    fp_sub(tmp,V,nX);fp_mul(tmp,rr,tmp);
    uint64_t s1j[6];fp_mul(s1j,S1,J);fp_add(s1j,s1j,s1j);
    fp_sub(nY,tmp,s1j);
    fp_add(tmp,p.Z,q.Z);fp_sqr(tmp,tmp);fp_sub(tmp,tmp,Z1Z1);fp_sub(tmp,tmp,Z2Z2);
    fp_mul(nZ,tmp,H);
    fp_copy(r.X,nX);fp_copy(r.Y,nY);fp_copy(r.Z,nZ);}
__device__ void jac_to_affine(uint64_t* out_x,uint64_t* out_y,const JacPoint& p) {
    if(jac_is_zero(p)){fp_set_zero(out_x);fp_set_zero(out_y);return;}
    uint64_t z_inv[6],z_inv2[6],z_inv3[6];
    fp_inv(z_inv,p.Z);fp_sqr(z_inv2,z_inv);fp_mul(z_inv3,z_inv2,z_inv);
    fp_mul(out_x,p.X,z_inv2);fp_mul(out_y,p.Y,z_inv3);}
static constexpr int WINDOW_BITS=8;
static constexpr int NUM_WINDOWS=(256+WINDOW_BITS-1)/WINDOW_BITS;
static constexpr int NUM_BUCKETS=(1<<WINDOW_BITS)-1;
static constexpr int CHUNK_SIZE=512;
__device__ __forceinline__ uint32_t extract_window(const Scalar256& s,int window_idx) {
    int bit_offset=window_idx*WINDOW_BITS;
    int limb_idx=bit_offset/64; int bit_in_limb=bit_offset%64;
    uint64_t val=(limb_idx<4)?(s.limbs[limb_idx]>>bit_in_limb):0;
    if(bit_in_limb>0&&limb_idx+1<4) val|=s.limbs[limb_idx+1]<<(64-bit_in_limb);
    return (uint32_t)(val&((1ULL<<WINDOW_BITS)-1));}
__global__ void chunk_accumulate_kernel(
    const G1Affine* __restrict__ points,const Scalar256* __restrict__ scalars,
    JacPoint* __restrict__ chunk_buckets,int N,int num_chunks) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int chunk_id=tid/NUM_WINDOWS; int window_idx=tid%NUM_WINDOWS;
    if(chunk_id>=num_chunks) return;
    int start=chunk_id*CHUNK_SIZE; int end=start+CHUNK_SIZE; if(end>N) end=N;
    JacPoint* my_buckets=&chunk_buckets[((size_t)chunk_id*NUM_WINDOWS+window_idx)*NUM_BUCKETS];
    for(int i=start;i<end;++i){
        uint32_t idx=extract_window(scalars[i],window_idx);
        if(idx==0) continue;
        jac_add_affine(my_buckets[idx-1],my_buckets[idx-1],points[i].x,points[i].y);}}
__global__ void tree_reduce_kernel(
    JacPoint* __restrict__ chunk_buckets,int num_active,int stride) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int num_pairs=num_active/(2*stride);
    int total_work=num_pairs*NUM_WINDOWS*NUM_BUCKETS;
    if(tid>=total_work) return;
    int pair_idx=tid/(NUM_WINDOWS*NUM_BUCKETS);
    int rem=tid%(NUM_WINDOWS*NUM_BUCKETS);
    int window_idx=rem/NUM_BUCKETS; int bucket_id=rem%NUM_BUCKETS;
    int c1=pair_idx*2*stride; int c2=c1+stride;
    size_t idx1=((size_t)c1*NUM_WINDOWS+window_idx)*NUM_BUCKETS+bucket_id;
    size_t idx2=((size_t)c2*NUM_WINDOWS+window_idx)*NUM_BUCKETS+bucket_id;
    JacPoint a=chunk_buckets[idx1]; JacPoint b=chunk_buckets[idx2];
    if(!jac_is_zero(b)){if(jac_is_zero(a)){chunk_buckets[idx1]=b;}else{jac_add(a,a,b);chunk_buckets[idx1]=a;}}}
__global__ void running_sum_kernel(
    JacPoint* __restrict__ final_buckets,JacPoint* __restrict__ window_sums,int num_windows) {
    int w=blockIdx.x*blockDim.x+threadIdx.x;
    if(w>=num_windows) return;
    JacPoint running;jac_set_zero(running);
    JacPoint total;jac_set_zero(total);
    JacPoint* wb=&final_buckets[(size_t)w*NUM_BUCKETS];
    for(int b=NUM_BUCKETS-1;b>=0;--b){
        if(!jac_is_zero(wb[b])) jac_add(running,running,wb[b]);
        if(!jac_is_zero(running)) jac_add(total,total,running);}
    window_sums[w]=total;}
__global__ void combine_windows_kernel(
    JacPoint* __restrict__ window_sums,G1Affine* __restrict__ out_q,int num_windows) {
    if(threadIdx.x!=0||blockIdx.x!=0) return;
    JacPoint result;jac_set_zero(result);
    for(int w=num_windows-1;w>=0;--w){
        for(int d=0;d<WINDOW_BITS;++d) jac_double(result,result);
        jac_add(result,result,window_sums[w]);}
    jac_to_affine(out_q->x,out_q->y,result);}
__global__ void zero_out_q_kernel(G1Affine* __restrict__ out_q) {
    if(threadIdx.x==0&&blockIdx.x==0) for(int i=0;i<6;++i){out_q->x[i]=0;out_q->y[i]=0;}}
} // namespace
void msm_bls12_381_g1(const G1Affine* points,const Scalar256* scalars,
    G1Affine* out_q,void* workspace,size_t workspace_bytes,int N,cudaStream_t stream) {
    if(!points||!scalars||!out_q||!workspace||N<=0){zero_out_q_kernel<<<1,1,0,stream>>>(out_q);return;}
    int num_chunks=(N+CHUNK_SIZE-1)/CHUNK_SIZE;
    size_t cb_sz=(size_t)num_chunks*NUM_WINDOWS*NUM_BUCKETS*sizeof(JacPoint);
    size_t ws_sz=(size_t)NUM_WINDOWS*sizeof(JacPoint);
    if(cb_sz+ws_sz>workspace_bytes){zero_out_q_kernel<<<1,1,0,stream>>>(out_q);return;}
    char* ws=(char*)workspace;
    JacPoint* chunk_buckets=(JacPoint*)ws; ws+=cb_sz;
    JacPoint* window_sums=(JacPoint*)ws;
    cudaMemsetAsync(chunk_buckets,0,cb_sz,stream);
    int total_threads=num_chunks*NUM_WINDOWS;
    int thr=128; int blk=(total_threads+thr-1)/thr;
    chunk_accumulate_kernel<<<blk,thr,0,stream>>>(points,scalars,chunk_buckets,N,num_chunks);
    for(int stride=1;stride<num_chunks;stride*=2){
        int num_pairs=num_chunks/(2*stride);
        int tw=num_pairs*NUM_WINDOWS*NUM_BUCKETS;
        if(tw<=0) break;
        int rb=(tw+255)/256;
        tree_reduce_kernel<<<rb,256,0,stream>>>(chunk_buckets,num_chunks,stride);}
    running_sum_kernel<<<1,32,0,stream>>>(chunk_buckets,window_sums,NUM_WINDOWS);
    combine_windows_kernel<<<1,1,0,stream>>>(window_sums,out_q,NUM_WINDOWS);}