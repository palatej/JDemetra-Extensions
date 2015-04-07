/*
 * Copyright 2013-2014 National Bank of Belgium
 * 
 * Licensed under the EUPL, Version 1.1 or – as soon they will be approved 
 * by the European Commission - subsequent versions of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 * 
 * http://ec.europa.eu/idabc/eupl
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and 
 * limitations under the Licence.
 */
package ec.tstoolkit.mdata;

import ec.tstoolkit.BaseException;
import ec.tstoolkit.data.DataBlock;
import ec.tstoolkit.data.DataBlockIterator;
import ec.tstoolkit.data.IReadDataBlock;
import ec.tstoolkit.data.ReadDataBlock;
import ec.tstoolkit.dstats.Chi2;
import ec.tstoolkit.dstats.F;
import ec.tstoolkit.dstats.IDistribution;
import ec.tstoolkit.dstats.ProbabilityType;
import ec.tstoolkit.maths.Complex;
import ec.tstoolkit.maths.matrices.EigenSystem;
import ec.tstoolkit.maths.matrices.IEigenSystem;
import ec.tstoolkit.maths.matrices.Matrix;
import ec.tstoolkit.maths.matrices.SymmetricMatrix;
import ec.tstoolkit.utilities.IntList;

/**
 *
 * @author Jean Palate, Frank Osaer
 */
public class Mahalanobis {

    private final IMultivariateDistribution m_dist;
    private double[] m_distances, m_stats, m_probs;
    private Matrix m_stdobs;

    /**
     * Creates a Mahalanobis object from a given distribution and from a set of
     * new data. The distribution is not necessarily the sample distribution of
     * the given data set No transformation is applied to the data. Users should
     * call the static logTransform method if they want to use log10
     * transformation of the data
     *
     * @param dist The multi-variate distribution
     * @param data The data
     */
    public Mahalanobis(final IMultivariateDistribution dist, final Matrix data) {
        m_dist = dist;
        compute(data);
    }

    /**
     * Creates a Mahalanobis object from a multi-variate data object)
     *
     * @param data The input. The moments and the dataset corresponding to that
     * object will be used to initialize the new object. No transformation is
     * applied to the data. Users should call the static logTransform method if
     * they want to use log10 transformation of the data
     */
    public Mahalanobis(final MultivariateData data) {
        m_dist = data.getSampleDistribution();
        compute(data);
    }

    private void compute(final MultivariateData data) {
        if (data == null) {
            throw new NullPointerException("Data cannot be null");
        }

        // get the covariance matrix
        Matrix sm = m_dist.getCovariances();
        IEigenSystem ies = EigenSystem.create(sm);

        // get the eigenvalues of S
        ies.setComputingEigenVectors(true);
        Complex[] evals = ies.getEigenValues();

        // compute S^(0.5) and inverse the resulting "matrix"
        double[] devals = new double[evals.length];
        for (int i = 0; i < evals.length; i++) {
            devals[i] = 1.0 / Math.sqrt(evals[i].re);
        }

        Matrix im = ies.getEigenVectors();
        Matrix dm = Matrix.diagonal(devals);

        // compute S2
        Matrix s2b = SymmetricMatrix.quadraticFormT(dm, im);

        // mahalanobis distance
        Matrix m = data.getMeanCorrectedData();

        m_stdobs = m.times(s2b);
        m_distances = new double[m_stdobs.getRowsCount()];

        for (int i = 0; i < m_distances.length; i++) {
            m_distances[i] = m_stdobs.row(i).nrm2();
        }
    }

    private void compute(final Matrix data) {
        if (data == null) {
            throw new NullPointerException("Data cannot be null");
        }
        if (data.getColumnsCount() != m_dist.getDim()) {
            throw new BaseException("Incompatible dataset");
        }

        // get the covariance matrix
        Matrix sm = m_dist.getCovariances();
        IEigenSystem ies = EigenSystem.create(sm);

        // get the eigenvalues of S
        ies.setComputingEigenVectors(true);
        Complex[] evals = ies.getEigenValues();

        // compute S^(0.5) and inverse the resulting "matrix"
        double[] devals = new double[evals.length];
        for (int i = 0; i < evals.length; i++) {
            devals[i] = 1.0 / Math.sqrt(evals[i].re);
        }

        Matrix im = ies.getEigenVectors();
        Matrix dm = Matrix.diagonal(devals);

        // compute S2
        Matrix s2b = SymmetricMatrix.quadraticFormT(dm, im);

        // mahalanobis distance
        Matrix m = data.clone();
        IReadDataBlock means = m_dist.getMeans();
        for (int i = 0; i < m_dist.getDim(); ++i) {
            m.column(i).sub(means.get(i));
        }

        m_stdobs = m.times(s2b);
        m_distances = new double[m_stdobs.getRowsCount()];

        for (int i = 0; i < m_distances.length; i++) {
            m_distances[i] = m_stdobs.row(i).nrm2();
        }
    }

    private static int CHI_THRESHOLD = 100000;

    private void computeProbs() {
        m_stats = new double[m_distances.length];
        m_probs = new double[m_distances.length];

        int p = m_dist.getDim();
        if (m_dist instanceof ISampleMultivariateDistribution) {
            ISampleMultivariateDistribution D = (ISampleMultivariateDistribution) m_dist;
            int n = D.getSampleSize();
            if (n <= CHI_THRESHOLD) {
                F f = new F();
                f.setDFNum(p);
                f.setDFDenom(n - p);
                double dn=n;
                double c = dn * (dn - p);
                c /= p * (dn - 1) * (dn + 1);
                for (int i = 0; i < m_distances.length; i++) {
                    double val = m_distances[i];
                    val = val * val * c;
                    m_stats[i] = val;
                    m_probs[i] = f.getProbability(val, ProbabilityType.Lower);
                }
                return;
            }
        }

        Chi2 chi = new Chi2();
        chi.setDegreesofFreedom(p);
        for (int i = 0; i < m_distances.length; i++) {
            double val = m_distances[i];
            val = val * val;
            m_stats[i] = val;
            m_probs[i] = chi.getProbability(val, ProbabilityType.Lower);
        }
    }

    /**
     * Gets the statistics corresponding to the distribution.
     *
     * @return Read-only array of doubles, in an order corresponding to the
     * order of the original figures
     */
    public IReadDataBlock getStats() {
        if (m_stats == null) {
            computeProbs();
        }
        return new ReadDataBlock(m_stats);
    }

    public IReadDataBlock getProbabilities() {
        if (m_stats == null) {
            computeProbs();
        }
        return new ReadDataBlock(m_probs);
    }

    /**
     * Gets the distribution behind the distance
     *
     * @return Returns either a Chi2 or a F distribution, following the
     * availability of the sample size
     */
    public IDistribution getDistribution() {
        int p = m_dist.getDim();
        if (m_dist instanceof ISampleMultivariateDistribution) {
            ISampleMultivariateDistribution D = (ISampleMultivariateDistribution) m_dist;
            int n = D.getSampleSize();
            if (n <= CHI_THRESHOLD) {
                F f = new F();
                f.setDFNum(p);
                f.setDFDenom(n - p);
                return f;
            }
        }

        Chi2 chi = new Chi2();
        chi.setDegreesofFreedom(p);
        return chi;

    }

    /**
     * Gets the multivariate distribution
     *
     * @return
     */
    public IMultivariateDistribution getMultivariateDistribution() {
        return m_dist;
    }

    /**
     * Gets the distances in an order corresponding to the original data.
     *
     * @return
     */
    public IReadDataBlock getDistances() {
        return new ReadDataBlock(m_distances);
    }

    /**
     * Retrieves the positions (0-based) of the observations that are outside
     * the ellipse corresponding to the given probability.
     *
     * @param prob The threshold
     * @return The 0-based positions of the outliers. Never null.
     */
    public int[] searchOutliers(double prob) {
        IntList list = new IntList();
        // Distance corresponding to the probability.
        IDistribution dist = getDistribution();
        double d = dist.getProbabilityInverse(prob, ProbabilityType.Lower);
        if (m_dist instanceof ISampleMultivariateDistribution) {
            ISampleMultivariateDistribution D = (ISampleMultivariateDistribution) m_dist;
            int n = D.getSampleSize();
            if (n <= CHI_THRESHOLD) {
                double p=D.getDim();
                double dn=n;
                double c = dn * (dn - p);
                c /= p * (dn - 1) * (dn + 1);
                d/=c;
            }
        }
        d=Math.sqrt(d);
        for (int i = 0; i < m_distances.length; ++i) {
            if (m_distances[i] > d) {
                list.add(i);
            }
        }
        return list.toArray();
    }

    /**
     * Gets the normalized observations
     *
     * @return
     */
    public Matrix getNormalizedObservations() {
        return m_stdobs;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="maxrank"></param>
    /// <returns></returns>
    public static int[] rankFrequency(Matrix im, int maxrank) {
        int[] rf = new int[maxrank + 1];
        int[] ranks = ranks(im);
        for (int i = 0; i < ranks.length; i++) {
            if (ranks[i] <= maxrank) {
                rf[ranks[i]]++;
            }
        }
        return rf;
    }

    public static int[] ranks(Matrix im) {
        int nr = im.getRowsCount(), nc = im.getColumnsCount();
        DataBlockIterator rows = im.rows();
        DataBlock row = rows.getData();
        int[] data = new int[nr];
        int i = 0;
        do {
            double d = row.sum() / nc;
            data[i++] = (int) (Math.floor(Math.log10(d)));
        } while (rows.next());
        return data;
    }

    public static void logTransform(double[] data) {
        for (int i = 0; i < data.length; ++i) {
            if (data[i] <= 0.0) {
                throw new BaseException("The data contains zeroes or negative values. No Log10 transform is allowed");
            }
            data[i] = Math.log10(data[i]);
        }
    }

    public static Matrix logTransform(Matrix m) {
        Matrix mc = m.clone();
        double[] vals = mc.internalStorage();
        logTransform(vals);
        return mc;
    }

}
