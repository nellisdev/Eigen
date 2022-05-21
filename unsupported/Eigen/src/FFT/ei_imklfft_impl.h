// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. 
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "./InternalHeaderCheck.h"

#include <mkl_dfti.h>

#include <complex>
#include <stdexcept>

namespace Eigen {
    namespace internal {

      template <typename T>
      inline
      T * fftimkl_cast(const T * p)
      {
          return const_cast<T*>(p);
      }


      inline
      MKL_Complex16 * fftimkl_cast(const std::complex<double> *p)
      {
          return const_cast<MKL_Complex16*>(reinterpret_cast<const MKL_Complex16*>(p));
      }

      inline
      MKL_Complex8 * fftimkl_cast(const std::complex<float> *p)
      {
          return const_cast<MKL_Complex8*>(reinterpret_cast<const MKL_Complex8*>(p));
      }

      template <typename T>
      struct fftimkl_plan
      {
      };

      template <>
      struct fftimkl_plan<float>
      {
          typedef float scalar_type;
          typedef MKL_Complex8 complex_type;

          DFTI_DESCRIPTOR_HANDLE m_plan;
          MKL_LONG status;

          fftimkl_plan() :m_plan(0), status(0) {}
          ~fftimkl_plan() { if (m_plan) DftiFreeDescriptor(&m_plan); };

          inline
          void fwd(complex_type* dst, complex_type* src,int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
          void inv(complex_type* dst, complex_type* src,int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }

          inline
          void fwd(complex_type* dst, scalar_type* src,int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");
 
                  // Set CCE storage
                  status = DftiSetValue(m_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
          void inv(scalar_type* dst, complex_type* src,int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  // Set CCE storage
                  status = DftiSetValue(m_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }

          inline
          void fwd2(complex_type* dst, complex_type* src,int n0,int n1) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  MKL_LONG N[2]; N[0] = n0; N[1] = n1;
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, N);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
          void inv2(complex_type* dst, complex_type* src,int n0,int n1) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  MKL_LONG N[2]; N[0] = n0; N[1] = n1;
                  status = DftiCreateDescriptor(&m_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, N);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }
      };

      template <>
      struct fftimkl_plan<double>
      {
          typedef double scalar_type;
          typedef MKL_Complex16 complex_type;

          DFTI_DESCRIPTOR_HANDLE m_plan;
          MKL_LONG status;

          fftimkl_plan() :m_plan(0), status(0) {}
          ~fftimkl_plan() { if (m_plan) DftiFreeDescriptor(&m_plan); };

          inline
              void fwd(complex_type* dst, complex_type* src, int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
              void inv(complex_type* dst, complex_type* src, int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }

          inline
              void fwd(complex_type* dst, scalar_type* src, int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  // Set CCE storage
                  status = DftiSetValue(m_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
              void inv(scalar_type* dst, complex_type* src, int nfft) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)nfft);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  // Set CCE storage
                  status = DftiSetValue(m_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }

          inline
              void fwd2(complex_type* dst, complex_type* src, int n0, int n1) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  MKL_LONG N[2]; N[0] = n0; N[1] = n1;
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_COMPLEX, 2, N);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeForward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeForward failed.");
          }

          inline
              void inv2(complex_type* dst, complex_type* src, int n0, int n1) 
          {
              if (m_plan == 0) {
                  // Configure a Descriptor
                  MKL_LONG N[2]; N[0] = n0; N[1] = n1;
                  status = DftiCreateDescriptor(&m_plan, DFTI_DOUBLE, DFTI_COMPLEX, 2, N);
                  if (0 != status) throw std::runtime_error("DftiCreateDescriptor failed.");

                  status = DftiSetValue(m_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                  if (0 != status) throw std::runtime_error("DftiSetValue failed.");

                  status = DftiCommitDescriptor(m_plan);
                  if (0 != status) throw std::runtime_error("DftiCommitDescriptor failed.");
              }
              status = DftiComputeBackward(m_plan, src, dst);
              if (0 != status) throw std::runtime_error("DftiComputeBackward failed.");
          }
      };


      template <typename Scalar_>
      struct imklfft_impl
      {
          typedef Scalar_ Scalar;
          typedef std::complex<Scalar> Complex;

          inline
          void clear()
          {
            m_plans.clear();
          }

          // complex-to-complex forward FFT
          inline
          void fwd(Complex* dst,const Complex* src,int nfft)
          {
            get_plan(nfft,false,dst,src).fwd(fftimkl_cast(dst), fftimkl_cast(src),nfft);
          }

          // real-to-complex forward FFT
          inline
          void fwd(Complex* dst,const Scalar* src,int nfft)
          {
              get_plan(nfft,false,dst,src).fwd(fftimkl_cast(dst), fftimkl_cast(src) ,nfft);
          }

          // 2-d complex-to-complex
          inline
          void fwd2(Complex* dst, const Complex* src, int n0,int n1)
          {
              get_plan(n0,n1,false,dst,src).fwd2(fftimkl_cast(dst), fftimkl_cast(src) ,n0,n1);
          }

          // inverse complex-to-complex
          inline
          void inv(Complex* dst,const Complex* src,int nfft)
          {
            get_plan(nfft,true,dst,src).inv(fftimkl_cast(dst), fftimkl_cast(src),nfft);
          }

          // half-complex to scalar
          inline
          void inv(Scalar* dst,const Complex* src,int nfft)
          {
            get_plan(nfft,true,dst,src).inv(fftimkl_cast(dst), fftimkl_cast(src),nfft);
          }

          // 2-d complex-to-complex
          inline
          void inv2(Complex* dst, const Complex* src, int n0,int n1)
          {
            get_plan(n0,n1,true,dst,src).inv2(fftimkl_cast(dst), fftimkl_cast(src) ,n0,n1);
          }


      protected:
          typedef fftimkl_plan<Scalar> PlanData;

          typedef Eigen::numext::int64_t int64_t;

          typedef std::map<int64_t,PlanData> PlanMap;

          PlanMap m_plans;

          inline
          PlanData& get_plan(int nfft,bool inv,void* dst,const void* src)
          {
              int inverse = inv == true ? 1 : 0;
              int inplace = dst == src ? 1 : 0;
              int aligned = ((reinterpret_cast<size_t>(src) & 15) | (reinterpret_cast<size_t>(dst) & 15)) == 0 ? 1 : 0;
              int64_t key = ((nfft << 3) | (inverse << 2) | (inplace << 1) | aligned) << 1;
              return m_plans[key];
          }

          inline
          PlanData& get_plan(int n0,int n1,bool inv,void* dst,const void* src)
          {
              int inverse = inv == true ? 1 : 0;
              int inplace = (dst == src) ? 1 : 0;
              int aligned = ((reinterpret_cast<size_t>(src) & 15) | (reinterpret_cast<size_t>(dst) & 15)) == 0 ? 1 : 0;
              int64_t key = (((((int64_t)n0) << 30) | (n1 << 3) | (inverse << 2) | (inplace << 1) | aligned) << 1) + 1;
              return m_plans[key];
          }
      };

    }
}
