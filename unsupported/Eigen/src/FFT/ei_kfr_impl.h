// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. 
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "./InternalHeaderCheck.h"

//#define KFR_STD_COMPLEX

#include <kfr/dft.hpp>

#include <complex>

namespace Eigen {
    namespace internal {

        template <typename T>
        inline kfr::complex<T>* std_to_krf_complex_cast(std::complex<T>* p)
        {
            return reinterpret_cast<kfr::complex<T>*>(p);
        }

        template <typename T>
        inline const kfr::complex<T>* std_to_krf_complex_cast(const std::complex<T>* p)
        {
            return reinterpret_cast<const kfr::complex<T>*>(p);
        }

        template <typename T>
        inline T* fftkfr_cast(const T* p)
        {
            return const_cast<T*>(p);
        }

        template <typename T>
        inline kfr::complex<T>* fftkfr_cast(const std::complex<T>* p)
        {
            return const_cast<kfr::complex<T>*>(reinterpret_cast<const kfr::complex<T>*>(p));
        }


        template <typename Scalar_>
        struct kfr_impl
        {
            using Scalar = Scalar_;
            using Complex = std::complex<Scalar>;

            inline void clear()
            {
                kfr::dft_cache::instance().clear();
            }

            // complex-to-complex forward FFT
            inline void fwd(Complex* dst, const Complex* src, int nfft)
            {
                kfr::dft_plan_ptr<Scalar> dft = kfr::dft_cache::instance().get(kfr::ctype_t<Scalar>(), nfft);
                kfr::univector<kfr::u8> temp(dft->temp_size);
                dft->execute(std_to_krf_complex_cast(dst), std_to_krf_complex_cast(src), temp.data());
            }

            // real-to-complex forward FFT
            inline void fwd(Complex* dst, const Scalar* src, int nfft)
            {
                kfr::dft_plan_real_ptr<Scalar> dft = kfr::dft_cache::instance().getreal(kfr::ctype_t<Scalar>(), nfft);
                kfr::univector<kfr::u8> temp(dft->temp_size);
                dft->execute(std_to_krf_complex_cast(dst), src, temp.data());
            }

            // 2-d complex-to-complex
            inline void fwd2(Complex* dst, const Complex* src, int n0, int n1)
            {
                throw std::logic_error("fwd2 not implemented.");
            }

            // inverse complex-to-complex
            inline void inv(Complex* dst, const Complex* src, int nfft)
            {
                kfr::dft_plan_ptr<Scalar> dft = kfr::dft_cache::instance().get(kfr::ctype_t<Scalar>(), nfft);
                kfr::univector<kfr::u8> temp(dft->temp_size);
                dft->execute(std_to_krf_complex_cast(dst), std_to_krf_complex_cast(src), temp.data(), kfr::ctrue);
            }

            // half-complex to scalar
            inline void inv(Scalar* dst, const Complex* src, int nfft)
            {
                kfr::dft_plan_real_ptr<Scalar> dft = kfr::dft_cache::instance().getreal(kfr::ctype_t<Scalar>(), nfft);
                kfr::univector<kfr::u8> temp(dft->temp_size);
                dft->execute(dst, std_to_krf_complex_cast(src), temp.data());
            }

            // 2-d complex-to-complex
            inline void inv2(Complex* dst, const Complex* src, int n0, int n1)
            {
                throw std::logic_error("fwd2 not implemented.");
            }
        };

    }
}