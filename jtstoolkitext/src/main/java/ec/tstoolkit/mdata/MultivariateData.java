/*
 * Copyright 2015 National Bank of Belgium
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
import ec.tstoolkit.data.DescriptiveStatistics;
import ec.tstoolkit.data.IReadDataBlock;
import ec.tstoolkit.data.ReadDataBlock;
import ec.tstoolkit.maths.matrices.Matrix;
import ec.tstoolkit.maths.matrices.SubMatrix;
import ec.tstoolkit.maths.matrices.SymmetricMatrix;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a collection of data series. Several statistics can be computed on
 * a collection of series: correlations, covariances, ... The class can be
 * interpreted as a matrix where the columns are the data series. Series must
 * have equal length
 *
 * @author Jean Palate, Frank Osaer
 */
public class MultivariateData {

    public static class Factory {

        private final List<double[]> items_ = new ArrayList<>();
        private final int dim_;

        public Factory(int dim) {
            dim_ = dim;
        }

        public void add(double[] data) {
            if (data == null || data.length != dim_) {
                throw new java.lang.IllegalArgumentException("Invalid data in MultivarateData.Factory");
            }
            for (int i = 0; i < data.length; ++i) {
                if (!DescriptiveStatistics.isFinite(data[i])) {
                    throw new java.lang.IllegalArgumentException("Invalid data in MultivarateData.Factory");
                }
            }
            items_.add(data);
        }

        public void add(IReadDataBlock data) {
            if (data == null || data.getLength() != dim_) {
                throw new java.lang.IllegalArgumentException("Invalid data in MultivarateData.Factory");
            }
            double[] d = new double[dim_];
            data.copyTo(d, 0);
            for (int i = 0; i < d.length; ++i) {
                if (!DescriptiveStatistics.isFinite(d[i])) {
                    throw new java.lang.IllegalArgumentException("Invalid data in MultivarateData.Factory");
                }
            }
            items_.add(d);
        }

        public void clear() {
            items_.clear();
        }

        public Matrix create() {
            if (items_.size() < dim_) {
                return null;
            }
            Matrix M = new Matrix(items_.size(), dim_);
            DataBlockIterator rows = M.rows();
            DataBlock row = rows.getData();
            int irow = 0;
            do {
                row.copyFrom(items_.get(irow++), 0);
            } while (rows.next());
            return M;
        }
    }

    private Matrix matrix_;
    private SampleMultivariateDistribution dist_;
    private final boolean adjust_;

    /**
     * Default constructor
     */
    public MultivariateData() {
        adjust_ = true;
    }

    /**
     * Default constructor
     *
     * @param nadjust Adjust the degrees of freedom for the variances
     */
    public MultivariateData(boolean nadjust) {
        adjust_ = nadjust;
    }

    /**
     * The matrix is copied.
     *
     * @param matrix
     */
    public void set(Matrix matrix) {
        matrix_ = matrix.clone();
        initStats();
    }

    /**
     * Constructor. The sub-matrix is copied
     *
     * @param matrix
     */
    public void set(SubMatrix matrix) {
        matrix_ = new Matrix(matrix);
        initStats();
    }

    public SampleMultivariateDistribution getSampleDistribution() {
        return dist_;
    }


    /**
     * Returns the data.
     *
     * @return The normalised data
     */
    public Matrix normalizedData() {
        Matrix m = matrix_.clone();
        int i = 0;
        DataBlockIterator cols = m.columns();
        DataBlock col = cols.getData();
        do {
            col.div(dist_.getStdError(i++));
        } while (cols.next());
        return m;
    }

    /**
     * Returns the data.
     *
     * @return The normalised data
     */
    public Matrix getMeanCorrectedData() {
        return matrix_;
    }

    private void initStats() {
        if (matrix_ == null) {
            throw new NullPointerException("The matrix reference cannot be null");
        }

        int n = matrix_.getColumnsCount(), m = matrix_.getRowsCount();
        double[] means = new double[n];
        double[] stdevs = new double[n];

        DataBlockIterator cols = matrix_.columns();
        DataBlock col = cols.getData();
        Matrix cov=new Matrix(n, n);

        double c = m - (adjust_ ? 1 : 0);
        int cur = 0;
        do {
            double a = col.sum() / m;
            col.sub(a);
            means[cur] = a;
            double var;
            var = col.ssq();
            var /= c;
            cov.set(cur, cur, var);
            double e = Math.sqrt(var);
            stdevs[cur++] = e;
        } while (cols.next());
        for (int i=0; i<n; ++i){
            DataBlock ci=matrix_.column(i);
            for (int j=0; j<i; ++j){
                double v=ci.dot(matrix_.column(j))/c;
                cov.set(i, j, v);
                cov.set(j, i, v);
            }
        }
        
        dist_=new SampleMultivariateDistribution(means, stdevs, cov, m);
    }

}
