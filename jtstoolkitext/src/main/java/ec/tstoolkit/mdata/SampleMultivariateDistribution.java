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
import ec.tstoolkit.design.Immutable;
import ec.tstoolkit.maths.matrices.Matrix;

/**
 *
 * @author Jean Palate
 */
@Immutable
public class SampleMultivariateDistribution extends MultivariateDistribution implements ISampleMultivariateDistribution {

    private final int size_;

    public SampleMultivariateDistribution(double[] means, Matrix cov, int sampleSize) {
        super(means, cov);
        size_ = sampleSize;
    }

    public SampleMultivariateDistribution(double[] means, double[] err, Matrix cov, int sampleSize) {
        super(means, err, cov);
        size_ = sampleSize;
    }

    public SampleMultivariateDistribution(IReadDataBlock means, Matrix cov, int sampleSize) {
        super(means, cov);
        size_ = sampleSize;
    }
    
    @Override
    public int getSampleSize() {
        return size_;
    }

}
