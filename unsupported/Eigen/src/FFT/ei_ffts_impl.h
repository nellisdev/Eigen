// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "./InternalHeaderCheck.h"

#include "ffts.h"
#include "types.h"

#include <complex>

namespace Eigen {
    namespace internal {

        // ffts_cpx_* is compatible with std::complex
        // This assumes std::complex<T> layout is array of size 2 with real,imag

        template <typename T>
        inline T* ffts_cast(const T* p)
        {
            return const_cast<T*>(p);
        }

        inline ffts_cpx_32f* ffts_cast(const std::complex<float>* p)
        {
            return const_cast<ffts_cpx_32f*>(reinterpret_cast<const ffts_cpx_32f*>(p));
        }

        inline ffts_cpx_64f* ffts_cast(const std::complex<double>* p)
        {
            return const_cast<ffts_cpx_64f*>(reinterpret_cast<const ffts_cpx_64f*>(p));
        }

        inline std::complex<float>* ffts_cast(const ffts_cpx_32f* p)
        {
            return const_cast<std::complex<float>*>(reinterpret_cast<const std::complex<float>*>(p));
        }

        inline std::complex<double>* ffts_cast(const ffts_cpx_64f* p)
        {
            return const_cast<std::complex<double>*>(reinterpret_cast<const std::complex<double>*>(p));
        }

        template <typename T>
        struct ffts_plan {};

        template <>
        struct ffts_plan<float>
        {
            typedef float scalar_type;
            typedef ffts_cpx_32f complex_type;

            ffts_plan_t* m_plan;
            ffts_plan() : m_plan(0) {}
            ~ffts_plan()
            {
                if (m_plan) ffts_free(m_plan);
            }

            inline void fwd(complex_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d(nfft, FFTS_FORWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void inv(complex_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d(nfft, FFTS_BACKWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void fwd(complex_type* dst, const scalar_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_real(nfft, FFTS_FORWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void inv(scalar_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_real(nfft, FFTS_BACKWARD);
                ffts_execute(m_plan, src, dst);
            }

            inline void fwd2(complex_type* dst, const complex_type* src, int n0, int n1)
            {
                if (m_plan == 0) m_plan = ffts_init_2d(n0, n1, FFTS_FORWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void inv2(complex_type* dst, const complex_type* src, int n0, int n1)
            {
                if (m_plan == 0) m_plan = ffts_init_2d(n0, n1, FFTS_BACKWARD);
                ffts_execute(m_plan, src, dst);
            }
        };

        template <>
        struct ffts_plan<double>
        {
            typedef double scalar_type;
            typedef ffts_cpx_64f complex_type;

            ffts_plan_t* m_plan;
            ffts_plan() : m_plan(0) {}
            ~ffts_plan()
            {
                if (m_plan) ffts_free(m_plan);
            }

            inline void fwd(complex_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_64f(nfft, FFTS_FORWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void inv(complex_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_64f(nfft, FFTS_BACKWARD);
                ffts_execute(m_plan, src, dst);
            }
            inline void fwd(complex_type* dst, scalar_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_64f(nfft, FFTS_FORWARD);
                Eigen::ArrayXcd src1 = Eigen::Map<Eigen::ArrayXd>(src, nfft).cast<std::complex<double>>();
                complex_type* src2 = ffts_cast(src1.data());

                ffts_execute(m_plan, src2, dst);
                // ffts_execute(m_plan, src, dst);
            }
            inline void inv(scalar_type* dst, const complex_type* src, int nfft)
            {
                if (m_plan == 0) m_plan = ffts_init_1d_64f(nfft, FFTS_BACKWARD);

                Eigen::Map<Eigen::ArrayXd> dest(dst, nfft);

                Eigen::ArrayXcd dst1 = dest.cast<std::complex<double>>();
                complex_type* dst2 = ffts_cast(dst1.data());

                ffts_execute(m_plan, src, dst2);
                Eigen::Map<Eigen::ArrayXcd> dst3(ffts_cast(dst2), nfft);
                dest = dst3.real();
            }

            inline void fwd2(complex_type* dst, const complex_type* src, int n0, int n1)
            {
                throw std::logic_error("fwd2 not implemented.");

                /*if (m_plan == 0) m_plan = ffts_init_2d(n0, n1, FFTS_FORWARD);
                ffts_execute(m_plan, src, dst);*/
            }
            inline void inv2(complex_type* dst, const complex_type* src, int n0, int n1)
            {
                throw std::logic_error("inv2 not implemented.");

                /*if (m_plan == 0) m_plan = ffts_init_2d(n0, n1, FFTS_BACKWARD);
                ffts_execute(m_plan, src, dst);*/
            }
        };

        template <typename Scalar_>
        struct ffts_impl
        {
            typedef Scalar_ Scalar;
            typedef std::complex<Scalar> Complex;

            inline void clear() { m_plans.clear(); }

            // complex-to-complex forward FFT
            inline void fwd(Complex* dst, const Complex* src, int nfft)
            {
                get_plan(nfft, false, dst, src).fwd(ffts_cast(dst), ffts_cast(src), nfft);
            }

            // real-to-complex forward FFT
            inline void fwd(Complex* dst, const Scalar* src, int nfft)
            {
                get_plan(nfft, false, dst, src).fwd(ffts_cast(dst), ffts_cast(src), nfft);
            }

            // 2-d complex-to-complex
            inline void fwd2(Complex* dst, const Complex* src, int n0, int n1)
            {
                get_plan(n0, n1, false, dst, src).fwd2(ffts_cast(dst), ffts_cast(src), n0, n1);
            }

            // inverse complex-to-complex
            inline void inv(Complex* dst, const Complex* src, int nfft)
            {
                get_plan(nfft, true, dst, src).inv(ffts_cast(dst), ffts_cast(src), nfft);
            }

            // half-complex to scalar
            inline void inv(Scalar* dst, const Complex* src, int nfft)
            {
                get_plan(nfft, true, dst, src).inv(ffts_cast(dst), ffts_cast(src), nfft);
            }

            // 2-d complex-to-complex
            inline void inv2(Complex* dst, const Complex* src, int n0, int n1)
            {
                get_plan(n0, n1, true, dst, src).inv2(ffts_cast(dst), ffts_cast(src), n0, n1);
            }

        protected:
            typedef ffts_plan<Scalar> PlanData;

            typedef std::map<int64_t, PlanData> PlanMap;

            PlanMap m_plans;

            inline PlanData& get_plan(int nfft, bool inv, void* dst, const void* src)
            {
                int inverse = inv == true ? 1 : 0;
                int inplace = dst == src ? 1 : 0;
                int aligned = ((reinterpret_cast<size_t>(src) & 15) | (reinterpret_cast<size_t>(dst) & 15)) == 0 ? 1 : 0;
                int64_t key = ((nfft << 3) | (inverse << 2) | (inplace << 1) | aligned) << 1;
                return m_plans[key];
            }

            inline PlanData& get_plan(int n0, int n1, bool inv, void* dst, const void* src)
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
