/* Non-Negagive Least Squares Algorithm for Eigen.
 *
 * Copyright (C) 2021 Essex Edwards, <essex.edwards@gmail.com>
 * Copyright (C) 2013 Hannes Matuschek, hannes.matuschek at uni-potsdam.de
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/** \defgroup nnls Non-Negative Least Squares (NNLS) Module
 * This module provides a single class @c Eigen::NNLS implementing the NNLS algorithm.
 * The algorithm is described in "SOLVING LEAST SQUARES PROBLEMS", by Charles L. Lawson and
 * Richard J. Hanson, Prentice-Hall, 1974 and solves optimization problems of the form
 *
 * \f[ \min \left\Vert Ax-b\right\Vert_2^2\quad s.t.\, x\ge 0\,.\f]
 *
 * The algorithm solves the constrained least-squares problem above by iteratively improving
 * an estimate of which constraints are active (elements of \f$x\f$ equal to zero)
 * and which constraints are inactive (elements of \f$x\f$ greater than zero).
 * Each iteration, an unconstrained least-squares problem solves for the
 * components of \f$x\f$ in the (estimated) inactive set and the sets are updated.
 * The unconstrained least-squares problem minimizes \f$\left\Vert A^Px^P-b\right\Vert_2^2\f$,
 * where \f$A^P\f$ is a matrix formed by selecting all columns of A which are
 * in the inactive set \f$P\f$.
 *
 */

#ifndef EIGEN_NNLS_H
#define EIGEN_NNLS_H

#include "../../Eigen/Core"
#include "../../Eigen/QR"

#include <limits>

namespace Eigen {

/** \ingroup nnls
 * \class NNLS
 * \brief Implementation of the Non-Negative Least Squares (NNLS) algorithm.
 * \tparam MatrixType The type of the system matrix \f$A\f$.
 *
 * This class implements the NNLS algorithm as described in "SOLVING LEAST SQUARES PROBLEMS",
 * Charles L. Lawson and Richard J. Hanson, Prentice-Hall, 1974. This algorithm solves a least
 * squares problem iteratively and ensures that the solution is non-negative. I.e.
 *
 * \f[ \min \left\Vert Ax-b\right\Vert_2^2\quad s.t.\, x\ge 0 \f]
 *
 * The algorithm solves the constrained least-squares problem above by iteratively improving
 * an estimate of which constraints are active (elements of \f$x\f$ equal to zero)
 * and which constraints are inactive (elements of \f$x\f$ greater than zero).
 * Each iteration, an unconstrained least-squares problem solves for
 * the components of \f$x\f$ in the (estimated) inactive set and the sets are updated.
 * The unconstrained least-squares problem minimizes \f$\left\Vert A^Px^P-b\right\Vert_2^2\f$,
 * where \f$A^P\f$ is a matrix formed by selecting all columns of A which are
 * in the inactive set \f$P\f$.
 *
 * See <a href="https://en.wikipedia.org/wiki/Non-negative_least_squares">the
 * wikipedia page on non-negative least squares</a> for more background information.
 *
 * \note Please note that it is possible to construct an NNLS problem for which the
 *       algorithm does not converge. In practice these cases are extremely rare.
 */
template <class MatrixType_>
class NNLS {
 public:
  typedef MatrixType_ MatrixType;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = MatrixType::Options,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::Index Index;

  /** Type of a row vector of the system matrix \f$A\f$. */
  typedef Matrix<Scalar, ColsAtCompileTime, 1> SolutionVectorType;
  /** Type of a column vector of the system matrix \f$A\f$. */
  typedef Matrix<Scalar, RowsAtCompileTime, 1> RhsVectorType;
  typedef PermutationMatrix<ColsAtCompileTime, ColsAtCompileTime, Index> PermutationType;
  typedef typename PermutationType::IndicesType IndicesType;

  /** */
  NNLS();

  /** \brief Constructs a NNLS sovler and initializes it with the given system matrix @c A.
   * \param A Specifies the system matrix.
   * \param max_iter Specifies the maximum number of iterations to solve the system.
   * \param tol Specifies the precision of the optimum.
   *        This is an absolute tolerance on the gradient of the Lagrangian, \f$A^T(Ax-b)-\lambda\f$
   *        (with Lagrange multipliers \f$\lambda\f$).
   */
  NNLS(const MatrixType &A, Index max_iter = -1, Scalar tol = NumTraits<Scalar>::dummy_precision());

  /** Initializes the solver with the matrix \a A for further solving NNLS problems.
   *
   * This function mostly initializes/computes the preconditioner. In the future
   * we might, for instance, implement column reordering for faster matrix vector products.
   */
  template <typename MatrixDerived>
  NNLS<MatrixType> &compute(const EigenBase<MatrixDerived> &A);

  /** \brief Solves the NNLS problem.
   *
   * The dimension of @c b must be equal to the number of rows of @c A, given to the constructor.
   *
   * \returns The approximate solution vector \f$ x \f$. Use info() to determine if the solve was a success or not.
   * \sa info()
   */
  const SolutionVectorType &solve(const RhsVectorType &b);

  /** \brief Returns the solution if a problem was solved.
   * If not, an uninitialized vector may be returned. */
  const SolutionVectorType &x() const { return x_; }

  /** \returns the tolerance threshold used by the stopping criteria.
   * \sa setTolerance()
   */
  Scalar tolerance() const { return tolerance_; }

  /** Sets the tolerance threshold used by the stopping criteria.
   *
   * This is an absolute tolerance on the gradient of the Lagrangian, \f$A^T(Ax-b)-\lambda\f$
   * (with Lagrange multipliers \f$\lambda\f$).
   */
  NNLS<MatrixType> &setTolerance(const Scalar &tolerance) {
    tolerance_ = tolerance;
    return *this;
  }

  /** \returns the max number of iterations.
   * It is either the value set by setMaxIterations or, by default, twice the number of columns of the matrix.
   */
  Index maxIterations() const { return max_iter_ < 0 ? 2 * A_.cols() : max_iter_; }

  /** Sets the max number of iterations.
   * Default is twice the number of columns of the matrix.
   * The algorithm requires at least k iterations to produce a solution vector with k non-zero entries.
   */
  NNLS<MatrixType> &setMaxIterations(Index maxIters) {
    max_iter_ = maxIters;
    return *this;
  }

  /** \returns the number of iterations (least-squares solves) performed during the last solve */
  Index iterations() const { return iterations_; }

  /** \returns Success if the iterations converged, and an error values otherwise. */
  ComputationInfo info() const { return info_; }

 private:
  /** \internal Searches for the index in Z with the largest value of @c v (\f$argmax v^P\f$) . */
  Index argmax_Z_(const SolutionVectorType &v) {
    const IndicesType &idxs = set_permutation.indices();
    Index m_idx = Np_;
    Scalar m = v(idxs(m_idx));
    for (Index i = (Np_ + 1); i < A_.cols(); i++) {
      Index idx = idxs(i);
      if (m < v(idx)) {
        m = v(idx);
        m_idx = i;
      }
    }
    return m_idx;
  }

  /** \internal Searches for the largest value in \f$v^Z\f$. */
  Scalar max_Z_(const SolutionVectorType &v) {
    const IndicesType &idxs = set_permutation.indices();
    Scalar m = v(idxs(Np_));
    for (Index i = (Np_ + 1); i < A_.cols(); i++) {
      Index idx = idxs(i);
      if (m < v(idx)) {
        m = v(idx);
      }
    }
    return m;
  }

  /** \internal Searches for the smallest value in \f$v^P\f$. */
  Scalar min_P_(const SolutionVectorType &v) {
    eigen_assert(Np_ > 0);
    const IndicesType &idxs = set_permutation.indices();
    Scalar m = v(idxs(0));
    for (Index i = 1; i < Np_; i++) {
      Index idx = idxs(i);
      if (m > v(idx)) {
        m = v(idx);
      }
    }
    return m;
  }

  /** \internal Adds the given index @c idx to the set P and updates the QR decomposition of \f$A^P\f$. */
  void addToP_(Index idx);

  /** \internal Removes the given index idx from the set P and updates the QR decomposition of \f$A^P\f$. */
  void removeFromP_(Index idx);

  /** \internal Solves the least-squares problem \f$\left\Vert y-A^Px\right\Vert_2^2\f$. */
  void solveLS_P_(const RhsVectorType &b);

 private:
  typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixAtAType;

  /** \internal Holds the maximum number of iterations for the NNLS algorithm.
   *  @c -1 means to use the default value. */
  Index max_iter_;
  /** \internal Holds the number of iterations. */
  Index iterations_;
  /** \internal Holds success/fail of the last solve. */
  ComputationInfo info_;
  /** \internal Size of the P (inactive) set. */
  Index Np_;
  /** \internal Accuracy of the algorithm w.r.t the optimality of the solution (gradient). */
  Scalar tolerance_;
  /** \internal The system matrix, a copy of the one given to the constructor. */
  MatrixType A_;
  /** \internal Precomputed product \f$A^TA\f$. */
  MatrixAtAType AtA_;
  /** \internal Will hold the solution. */
  SolutionVectorType x_;
  /** \internal Will hold the current gradient. */
  SolutionVectorType w_;
  /** \internal Will hold the partial solution. */
  SolutionVectorType y_;
  /** \internal Precomputed product \f$A^Tb\f$. */
  SolutionVectorType Atb_;
  /** \internal Holds the current permutation matrix partitioning the active and inactive sets.
   * The first @c Np_ columns form the inactive set P and the rest the active set Z. */
  PermutationType set_permutation;
  /** \internal QR decomposition to solve the (inactive) sub system (together with @c qrCoeffs_). */
  MatrixType QR_;
  /** \internal QR decomposition to solve the (inactive) sub system (together with @c QR_). */
  SolutionVectorType qrCoeffs_;
  /** \internal Some workspace for QR decomposition. */
  SolutionVectorType tempRowVector_;
  RhsVectorType tempColVector_;
};

/* ********************************************************************************************
 * Implementation
 * ******************************************************************************************** */

namespace internal {

/** \internal
 * Basically a modified copy of @c Eigen::internal::householder_qr_inplace_unblocked that
 * performs a rank-1 update of the QR matrix in compact storage. This function assumes, that
 * the first @c k-1 columns of the matrix @c mat contain the QR decomposition of \f$A^P\f$ up to
 * column k-1. Then the QR decomposition of the k-th column (given by @c newColumn) is computed by
 * applying the k-1 Householder projectors on it and finally compute the projector \f$H_k\f$ of
 * it. On exit the matrix @c mat and the vector @c hCoeffs contain the QR decomposition of the
 * first k columns of \f$A^P\f$. */
template <typename MatrixQR, typename HCoeffs, typename VectorQR>
void nnls_householder_qr_inplace_update(MatrixQR &mat, HCoeffs &hCoeffs, const VectorQR &newColumn,
                                        typename MatrixQR::Index k, typename MatrixQR::Scalar *tempData = 0) {
  typedef typename MatrixQR::Index Index;
  typedef typename MatrixQR::Scalar Scalar;
  typedef typename MatrixQR::RealScalar RealScalar;
  Index rows = mat.rows();

  eigen_assert(k < mat.cols());
  eigen_assert(k < rows);
  eigen_assert(hCoeffs.size() == mat.cols());
  eigen_assert(newColumn.size() == rows);

  Matrix<Scalar, Dynamic, 1, ColMajor, MatrixQR::MaxColsAtCompileTime, 1> tempVector;
  if (tempData == 0) {
    tempVector.resize(mat.cols());
    tempData = tempVector.data();
  }

  // Store new column in mat at column k
  mat.col(k) = newColumn;
  // Apply H = H_1...H_{k-1} on newColumn (skip if k=0)
  for (Index i = 0; i < k; ++i) {
    Index remainingRows = rows - i;
    mat.col(k)
        .tail(remainingRows)
        .applyHouseholderOnTheLeft(mat.col(i).tail(remainingRows - 1), hCoeffs.coeffRef(i), tempData + i + 1);
  }
  // Construct Householder projector in-place in column k
  RealScalar beta;
  mat.col(k).tail(rows - k).makeHouseholderInPlace(hCoeffs.coeffRef(k), beta);
  mat.coeffRef(k, k) = beta;
}

}  // namespace internal

template <typename MatrixType>
NNLS<MatrixType>::NNLS()
    : max_iter_(-1),
      iterations_(0),
      info_(ComputationInfo::InvalidInput),
      Np_(0),
      tolerance_(NumTraits<Scalar>::dummy_precision()) {}

template <typename MatrixType>
NNLS<MatrixType>::NNLS(const MatrixType &A, Index max_iter, Scalar tol) : max_iter_(max_iter), tolerance_(tol) {
  compute(A);
}

template <typename MatrixType>
template <typename MatrixDerived>
NNLS<MatrixType> &NNLS<MatrixType>::compute(const EigenBase<MatrixDerived> &A) {
  // Ensure Scalar type is real. The non-negativity constraint doesn't obviously extend to complex numbers.
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL);

  // max_iter_: unchanged
  iterations_ = 0;
  info_ = ComputationInfo::Success;
  Np_ = 0;
  // tolerance: unchanged
  A_ = A.derived();
  AtA_.noalias() = A_.transpose() * A_;
  x_.resize(A_.cols());
  w_.resize(A_.cols());
  y_.resize(A_.cols());
  Atb_.resize(A_.cols());
  set_permutation.resize(A_.cols());
  QR_.resize(A_.rows(), A_.cols());
  qrCoeffs_.resize(A_.cols());
  tempRowVector_.resize(A_.cols());
  tempColVector_.resize(A_.rows());

  return *this;
}

template <typename MatrixType>
const typename NNLS<MatrixType>::SolutionVectorType &NNLS<MatrixType>::solve(const RhsVectorType &b) {
  // Initialize solver
  iterations_ = 0;
  info_ = ComputationInfo::NumericalIssue;
  x_.setZero();

  // Together with Np_, P separates the space of coefficients into a active (Z) and inactive (P)
  // set. The first Np_ elements form the inactive set P and the remaining elements form the
  // active set Z.
  set_permutation.setIdentity();
  Np_ = 0;

  // Precompute A^T*b
  Atb_.noalias() = A_.transpose() * b;

  const Index maxIterations = this->maxIterations();

  // OUTER LOOP
  while (true) {
    // Update the gradient
    w_.noalias() = Atb_ - AtA_ * x_;

    // Check if system is solved:
    if ((A_.cols() == Np_) || ((max_Z_(w_) - tolerance_) < 0)) {
      info_ = ComputationInfo::Success;
      return x_;
    }

    // We need a heuristic to choose the next parameter for the system update.
    // We choose the one with largest gradient.
    // Other heuristics are possible, but unimplemented.
    // Here is where those other options would go.
    addToP_(argmax_Z_(w_));

    // INNER LOOP
    while (true) {
      // Check if max. number of iterations is reached
      if (iterations_ >= maxIterations) {
        info_ = ComputationInfo::NoConvergence;
        return x_;
      }

      // Solve least-squares problem in P only,
      // this step is rather trivial as addToset_permutation & removeFromset_permutation
      // updates the QR decomposition of A^P.
      solveLS_P_(b);
      ++iterations_;  // The solve is expensive, so that is what we count as an iteration.

      // Check feasability...
      bool feasible = true;
      Scalar alpha = NumTraits<Scalar>::highest();
      Index remIdx = -1;
      for (Index i = 0; i < Np_; i++) {
        Index idx = set_permutation.indices()(i);
        if (y_(idx) < 0) {
          // t should always be in [0,1].
          Scalar t = -x_(idx) / (y_(idx) - x_(idx));
          if (alpha > t) {
            alpha = t;
            remIdx = i;
            feasible = false;
          }
        }
      }
      eigen_assert(feasible || 0 <= remIdx);

      // If solution is feasible, exit to outer loop
      if (feasible) {
        x_ = y_;
        break;
      }

      // Infeasible solution -> interpolate to feasible one
      for (Index i = 0; i < Np_; i++) {
        Index idx = set_permutation.indices()(i);
        x_(idx) += alpha * (y_(idx) - x_(idx));
      }

      // Remove these indices from P and update QR decomposition
      removeFromP_(remIdx);
    }
  }
}

template <typename MatrixType>
void NNLS<MatrixType>::addToP_(Index idx) {
  // Update permutation matrix:
  IndicesType &idxs = set_permutation.indices();

  std::swap(idxs(idx), idxs(Np_));
  Np_++;

  // Perform rank-1 update of the QR decomposition stored in QR_ & qrCoeff_
  internal::nnls_householder_qr_inplace_update(QR_, qrCoeffs_, A_.col(idxs(Np_ - 1)), Np_ - 1, tempRowVector_.data());
}

template <typename MatrixType>
void NNLS<MatrixType>::removeFromP_(Index idx) {
  // swap index with last inactive one & reduce number of inactive columns
  std::swap(set_permutation.indices()(idx), set_permutation.indices()(Np_ - 1));
  Np_--;
  // Update QR decomposition starting from the removed index up to the end [idx, ..., Np_]
  for (Index i = idx; i < Np_; i++) {
    Index col = set_permutation.indices()(i);
    internal::nnls_householder_qr_inplace_update(QR_, qrCoeffs_, A_.col(col), i, tempRowVector_.data());
  }
}

template <typename MatrixType>
void NNLS<MatrixType>::solveLS_P_(const RhsVectorType &b) {
  eigen_assert(Np_ > 0);

  tempColVector_ = b;

  // tmp(0:Np) := Q'*b
  // tmp(Np:end) := useless stuff we would rather not compute at all.
  tempColVector_.applyOnTheLeft(householderSequence(QR_.leftCols(Np_), qrCoeffs_.head(Np_)).transpose());

  // tmp(0:Np) := inv(R) * Q' * b = the least-squares solution for the inactive variables.
  QR_.topLeftCorner(Np_, Np_)                   //
      .template triangularView<Upper>()         //
      .solveInPlace(tempColVector_.head(Np_));  //

  // tmp(Np:A.rows()) := 0 = the value for the constrained variables.
  tempColVector_.middleRows(Np_, y_.size() - Np_).setZero();

  // Back permute y into original column order of A
  y_.noalias() = set_permutation * tempColVector_.head(y_.size());
}

}  // namespace Eigen

#endif  // EIGEN_NNLS_H
