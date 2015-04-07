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
import ec.tstoolkit.data.DescriptiveStatistics;
import ec.tstoolkit.data.IReadDataBlock;
import ec.tstoolkit.maths.matrices.Matrix;
import ec.tstoolkit.maths.matrices.SubMatrix;
import ec.tstoolkit.mdata.MultivariateData.Factory;
import java.util.Random;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Jean Palate
 */
public class MultivariateDataTest {

    private static final Matrix M;

    static {
        M = new Matrix(10000, 10);
        M.randomize(0);
    }

    public MultivariateDataTest() {
    }

    @Test
    public void testSet_Matrix() {
        Matrix matrix = M.clone();
        MultivariateData instance = new MultivariateData();
        instance.set(matrix);
    }

    @Test
    public void testSet_SubMatrix() {
        SubMatrix matrix = M.subMatrix();
        MultivariateData instance = new MultivariateData();
        instance.set(matrix);
    }

    @Test
    public void testCorrelations() {
        Matrix matrix = M.clone();
        matrix.add(25);
        MultivariateData instance1 = new MultivariateData(true);
        instance1.set(matrix);
        MultivariateData instance2 = new MultivariateData(true);
        SubMatrix submatrix = M.subMatrix();
        instance2.set(submatrix);
        Matrix result1 = instance1.getSampleDistribution().getCorrelations();
        Matrix result2 = instance2.getSampleDistribution().getCorrelations();
        assertTrue(result1.minus(result2).nrm2() < 1e-9);
    }

    @Test
    public void testCovariances() {
        Matrix matrix = M.clone();
        matrix.add(25);
        MultivariateData instance1 = new MultivariateData();
        instance1.set(matrix);
        MultivariateData instance2 = new MultivariateData();
        SubMatrix submatrix = M.subMatrix();
        instance2.set(submatrix);
        Matrix result1 = instance1.getSampleDistribution().getCovariances();
        Matrix result2 = instance2.getSampleDistribution().getCovariances();
        assertTrue(result1.minus(result2).nrm2() < 1e-9);
    }

    @Test
    public void testMean() {
        Matrix matrix = M.clone();
        matrix.add(25);
        MultivariateData instance1 = new MultivariateData(false);
        instance1.set(matrix);
        MultivariateData instance2 = new MultivariateData(false);
        SubMatrix submatrix = M.subMatrix();
        instance2.set(submatrix);
        for (int i = 0; i < M.getColumnsCount(); ++i) {
            assertTrue(Math.abs(instance1.getSampleDistribution().getMean(i) - instance2.getSampleDistribution().getMean(i) - 25) < 1e-9);
            DescriptiveStatistics stats = new DescriptiveStatistics(M.column(i));
            assertTrue(Math.abs(stats.getAverage() - instance2.getSampleDistribution().getMean(i)) < 1e-9);
        }
    }

    @Test
    public void testStdDev() {
        Matrix matrix = M.clone();
        matrix.mul(25);
        MultivariateData instance1 = new MultivariateData();
        instance1.set(matrix);
        MultivariateData instance2 = new MultivariateData();
        SubMatrix submatrix = M.subMatrix();
        instance2.set(submatrix);
        for (int i = 0; i < M.getColumnsCount(); ++i) {
            assertTrue(Math.abs(instance1.getSampleDistribution().getStdError(i) / 25 - instance2.getSampleDistribution().getStdError(i)) < 1e-9);
            DescriptiveStatistics stats = new DescriptiveStatistics(M.column(i));
            assertTrue(Math.abs(stats.getStdevDF(1) - instance2.getSampleDistribution().getStdError(i)) < 1e-9);
        }
    }

    @Test
    public void testVar() {
        Matrix matrix = M.clone();
        matrix.mul(2);
        MultivariateData instance1 = new MultivariateData(true);
        instance1.set(matrix);
        MultivariateData instance2 = new MultivariateData(true);
        SubMatrix submatrix = M.subMatrix();
        instance2.set(submatrix);
        for (int i = 0; i < M.getColumnsCount(); ++i) {
            assertTrue(Math.abs(instance1.getSampleDistribution().getVariance(i) / 4 - instance2.getSampleDistribution().getVariance(i)) < 1e-9);
            DescriptiveStatistics stats = new DescriptiveStatistics(M.column(i));
            assertTrue(Math.abs(stats.getVarDF(1) - instance2.getSampleDistribution().getVariance(i)) < 1e-9);
        }
    }

    @Test
    public void demoFactory() {
        // dimension of the Mahalanobis exercise. For instance
        int N = 2;
        // creation through the factory.
        // direct creation of the matrix is more efficient when the size is known.
        ec.tstoolkit.mdata.MultivariateData.Factory fac
                = new ec.tstoolkit.mdata.MultivariateData.Factory(N);
        DataBlock x = new DataBlock(N);
        Random rnd=new Random();
        for (int i = 0; i < 10000; ++i) {
            x.randomize();
            x.add(1);
            x.mul(rnd.nextDouble()*1000);
            fac.add(x);
        }
        MultivariateData md = new MultivariateData();
        Matrix M = fac.create();

        // Transforms the matrix if need be.
        M=ec.tstoolkit.mdata.Mahalanobis.logTransform(M);
        // Initializes the m. data
        md.set(M);

        // Gets the distribution of the sample. 
        ISampleMultivariateDistribution sampleDistribution = md.getSampleDistribution();

        // The data of the distribution should be stored in some database and reloaded later
        IReadDataBlock means = sampleDistribution.getMeans();
        Matrix cov = sampleDistribution.getCovariances();
        int sampleSize = sampleDistribution.getSampleSize();

        // ... later on ...
        ec.tstoolkit.mdata.SampleMultivariateDistribution refdist = new ec.tstoolkit.mdata.SampleMultivariateDistribution(means, cov, sampleSize);

        // Gets the new figures
        fac.clear();
        for (int i = 0; i < 150; ++i) {
            x.randomize();
            x.add(1);
            x.mul(rnd.nextDouble()*1000);
            fac.add(x);
        }
        // same transformation
        Matrix Cur = fac.create();
        Cur=ec.tstoolkit.mdata.Mahalanobis.logTransform(Cur);

        ec.tstoolkit.mdata.Mahalanobis mahalanobis = new ec.tstoolkit.mdata.Mahalanobis(refdist, Cur);

        // Gets the probabilities in an order corresponding to the order of the data in Cur
        IReadDataBlock probabilities = mahalanobis.getProbabilities();
        System.out.println(probabilities);
        
        // In another scenario, the distances could be computed directly on the original data set
        ec.tstoolkit.mdata.Mahalanobis mahalanobis2 = new ec.tstoolkit.mdata.Mahalanobis(md);

        // Gets the probabilities in an order corresponding to the order of the data in md
        probabilities = mahalanobis2.getProbabilities();
        System.out.println(probabilities.rextract(0, 20));
        int[] outliers = mahalanobis2.searchOutliers(0.99);
        assertTrue(outliers != null);
    }
}
