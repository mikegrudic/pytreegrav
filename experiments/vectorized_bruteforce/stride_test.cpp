// Bisect: which numba-ism stops clang from vectorizing the symmetric scatter loop?
//
// All three variants below vectorize (clang 16, -O3 -march=native -ffast-math, confirmed via
// -Rpass=loop-vectorize), which rules out runtime strides and branch-vs-ternary as the cause.
// The actual blocker turned out to be a runtime loop *lower bound* combined with a store --
// see i0_test.py in this directory.  Build:
//   clang++ -O3 -march=native -ffast-math -std=c++17 -c stride_test.cpp -o /dev/null \
//           -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
#include <cstdint>
#include <cstring>
#include <cmath>
// 1/sqrt(x) from multiplies only: magic-constant seed + 4 Newton steps (~7.6e-15 rel).
static inline double rsqrt4(double x){int64_t i;double y;std::memcpy(&i,&x,8);
 i=0x5FE6EB50C7B537A9LL-(i>>1);std::memcpy(&y,&i,8);double h=.5*x;
 y=y*(1.5-h*y*y);y=y*(1.5-h*y*y);y=y*(1.5-h*y*y);y=y*(1.5-h*y*y);return y;}

// V1: constant unit stride, ternary conditional.  The baseline. VECTORIZES (width 4).
__attribute__((noinline)) void v1(const double*px,const double*py,const double*pz,
 const double*m,double*ox,double*oy,double*oz,int i0,int n,double mi,double xi,double yi,double zi,double*r){
 double ax=0,ay=0,az=0;
 for(int j=i0;j<n;++j){double dx=px[j]-xi,dy=py[j]-yi,dz=pz[j]-zi;
  double r2=dx*dx+dy*dy+dz*dz;double ri=r2>0?rsqrt4(r2):0.0;double k=ri*ri*ri;
  ax+=k*m[j]*dx;ay+=k*m[j]*dy;az+=k*m[j]*dz;
  ox[j]-=k*mi*dx;oy[j]-=k*mi*dy;oz[j]-=k*mi*dz;}
 r[0]=ax;r[1]=ay;r[2]=az;}

// V2: strides are runtime values, as numba passes them in the array struct.
//     Tests whether an unprovable unit stride is the blocker. VECTORIZES (width 4) -- it is not.
__attribute__((noinline)) void v2(const double*px,const double*py,const double*pz,
 const double*m,double*ox,double*oy,double*oz,int i0,int n,double mi,double xi,double yi,double zi,
 int64_t sx,int64_t sy,int64_t sz,int64_t sm,int64_t so,double*r){
 double ax=0,ay=0,az=0;
 for(int j=i0;j<n;++j){double dx=px[j*sx]-xi,dy=py[j*sy]-yi,dz=pz[j*sz]-zi;
  double r2=dx*dx+dy*dy+dz*dz;double ri=r2>0?rsqrt4(r2):0.0;double k=ri*ri*ri;
  ax+=k*m[j*sm]*dx;ay+=k*m[j*sm]*dy;az+=k*m[j*sm]*dz;
  ox[j*so]-=k*mi*dx;oy[j*so]-=k*mi*dy;oz[j*so]-=k*mi*dz;}
 r[0]=ax;r[1]=ay;r[2]=az;}

// V3: the r2>0 test written as an if/else block rather than a ternary, in case numba's
//     control flow is the problem. VECTORIZES (width 4) -- it is not.
__attribute__((noinline)) void v3(const double*px,const double*py,const double*pz,
 const double*m,double*ox,double*oy,double*oz,int i0,int n,double mi,double xi,double yi,double zi,double*r){
 double ax=0,ay=0,az=0;
 for(int j=i0;j<n;++j){double dx=px[j]-xi,dy=py[j]-yi,dz=pz[j]-zi;
  double r2=dx*dx+dy*dy+dz*dz;double ri;
  if(r2>0){ri=rsqrt4(r2);}else{ri=0.0;}
  double k=ri*ri*ri;
  ax+=k*m[j]*dx;ay+=k*m[j]*dy;az+=k*m[j]*dz;
  ox[j]-=k*mi*dx;oy[j]-=k*mi*dy;oz[j]-=k*mi*dz;}
 r[0]=ax;r[1]=ay;r[2]=az;}
