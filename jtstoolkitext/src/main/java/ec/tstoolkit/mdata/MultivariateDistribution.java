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

import ec.tstoolkit.data.IReadDataBlock;
import ec.tstoolkit.data.ReadDataBlock;
import ec.tstoolkit.design.Immutable;
import ec.tstoolkit.maths.matrices.Matrix;
import ec.tstoolkit.maths.matrices.SymmetricMatrix;

/**
 *
 * @author Jean Palate
 */
@Immutable
public class MultivariateDistribution implements IMultivariateDistribution {

    private final double[] means_, stde_;
    private final Matrix cov_;

    public MultivariateDistribution(double[] means, Matrix cov) {
        means_ = means.clone();
        cov_ = cov.clone();
        stde_ = new double[means_.length];
        for (int i = 0; i < stde_.length; ++i) {
            stde_[i] = Math.sqrt(cov.get(i, i));
        }
    }

    MultivariateDistribution(double[] means, double[] stde, Matrix cov) {
        means_ = means;
        cov_ = cov;
        stde_ = stde;
    }

    public MultivariateDistribution(IReadDataBlock means, Matrix cov) {
        means_ = new double[means.getLength()];
        means.copyTo(means_, 0);
        cov_ = cov.clone();
        stde_ = new double[means_.length];
        for (int i = 0; i < stde_.length; ++i) {
            stde_[i] = Math.sqrt(cov.get(i, i));
        }
    }

    @Override
    public int getDim() {
        return means_.length;
    }

    @Override
    public IReadDataBlock getMeans() {
        return new ReadDataBlock(means_);
    }

    @Override
    public Matrix getCovariances() {
        return cov_;
    }

    public IReadDataBlock getStdErrors() {
        return new ReadDataBlock(stde_);
    }

    public Matrix getCorrelations() {
        Matrix cov = cov_.clone();
        int n = means_.length;
        cov.diagonal().set(1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; j++) {
                cov.mul(i, j, 1 / (stde_[i] * stde_[j]));
            }
        }
        SymmetricMatrix.fromLower(cov);
        return cov;

    }
    
    public double getMean(int idx){
        return means_[idx];
    }

    public double getVariance(int idx){
        return cov_.get(idx, idx);
    }
    
    public double getStdError(int idx){
        return stde_[idx];
    }
}
