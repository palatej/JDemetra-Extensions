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

import ec.tstoolkit.data.DataBlock;
import ec.tstoolkit.dstats.Chi2;
import ec.tstoolkit.dstats.Normal;
import ec.tstoolkit.maths.matrices.Matrix;
import ec.tstoolkit.random.IRandomNumberGenerator;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Jean Palate
 */
public class MahalanobisTest {

    private static final Matrix X;

    private static final double[] data = new double[]{
        265568, 482862, 977770, 212710, 26859, 31398, 66441, 5846505, 438414, 394801,
        10, 28, 57, 13, 1, 1, 2, 281, 19, 21
    };

    static {
        X = new Matrix(data, 10, 2);
    }
    private static final double[] dist = new double[]{
        0.842274476, 1.071921545, 1.055564166, 1.680404922, 1.422304003, 1.43745595,
        1.516486037, 2.24453048, 0.506485584, 0.721234073};

    private static final double[] prob = new double[]{
        0.241818508, 0.355465157, 0.34728068, 0.633495925, 0.524658119, 0.531505828,
        0.566349436, 0.807078513, 0.097266086, 0.185220433};

    public MahalanobisTest() {
    }

    @Test
    public void test1() {

        MultivariateData md = new MultivariateData();
        md.set(Mahalanobis.logTransform(X));

        Mahalanobis mh = new Mahalanobis(md);
        assertTrue(new DataBlock(mh.getDistances()).distance(new DataBlock(dist)) < 1e-6);
        assertTrue(new DataBlock(mh.getProbabilities()).distance(new DataBlock(prob)) < 1e-6);
    }

    @Test
    public void test2() {

        MultivariateData md = new MultivariateData();
        Matrix lx = Mahalanobis.logTransform(X);
        md.set(lx);

        Mahalanobis mh = new Mahalanobis(md.getSampleDistribution(), lx);
        assertTrue(new DataBlock(mh.getDistances()).distance(new DataBlock(dist)) < 1e-6);
        assertTrue(new DataBlock(mh.getProbabilities()).distance(new DataBlock(prob)) < 1e-6);
        int[] outliers = mh.searchOutliers(0.01);
        assertTrue(outliers != null && outliers.length == X.getRowsCount());
        outliers = mh.searchOutliers(0.99);
        assertTrue(outliers != null && outliers.length == 0);
    }

    @Test
    public void test3() {

        MultivariateData md = new MultivariateData();
        Matrix lx = Mahalanobis.logTransform(X);
        md.set(lx);
        // The moments can be retrieved as follows (for instance):
        ISampleMultivariateDistribution sampleDistribution = md.getSampleDistribution();
        System.out.println(sampleDistribution.getCovariances());
        System.out.println(sampleDistribution.getMeans());
        System.out.println(sampleDistribution.getSampleSize());

        MultivariateDistribution m = new MultivariateDistribution(md.getSampleDistribution().getMeans(), md.getSampleDistribution().getCovariances());

        Mahalanobis mh = new Mahalanobis(m, lx);
        assertTrue(mh.getDistribution() instanceof Chi2);
    }

    @Test
    public void test4() {
        int N = 5;
        ec.tstoolkit.mdata.MultivariateData.Factory fac = new ec.tstoolkit.mdata.MultivariateData.Factory(N);
        Normal n = new Normal();
        IRandomNumberGenerator rng = ec.tstoolkit.random.MersenneTwister.fromSystemNanoTime();
        for (int i = 0; i < 10000000; ++i) {
            double[] x = new double[N];
            for (int j = 0; j < N; ++j) {
                x[j] = n.random(rng);
            }
            fac.add(x);
        }
        Matrix m = fac.create();
        ec.tstoolkit.mdata.MultivariateData md = new ec.tstoolkit.mdata.MultivariateData();
        md.set(m);
        ec.tstoolkit.mdata.Mahalanobis mh = new ec.tstoolkit.mdata.Mahalanobis(md);
        int[] o = mh.searchOutliers(.99);
        System.out.println("SIMUL");
        System.out.println(o.length);
    }
}
