import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Random;

public class AddNoise {
    private final double[] td_clean;
    private final double[] td_dirty;

    private final int err_rate;  // error rate rate/1000
    private final double err_range;  // error range
    private final int err_length;  // error rate rate/1000

    private final double td_range;  // time series data range
    private final Random random;

    public AddNoise(double[] td_clean, int rate, double range, int length, int seed) throws Exception {
        this.td_clean = td_clean;
        this.td_dirty = new double[td_clean.length];

        this.err_rate = rate;
        this.err_range = range;
        this.err_length = length;
        this.random = new Random(seed);

        this.td_range = calRange();

        addNoise();
    }

    private double calRange() {  // data range
        double v_min = Double.MAX_VALUE, v_max = Double.MIN_VALUE;
        for (double value : this.td_clean) {
            if (value < v_min) v_min = value;
            if (value > v_max) v_max = value;
        }
        return v_max - v_min;
    }

    private void addNoise() {
        int err_flag = 0;
        double err_range_now = 0.0;
        for (int i = 0; i < td_clean.length; i++) {
            int i1 = random.nextInt(1000);
            if (i1 < this.err_rate) {
                err_flag = err_length;
                err_range_now = random.nextGaussian() * this.td_range * this.err_range;
            }

            if (err_flag > 0) {
                --err_flag;
                double new_value = td_clean[i] + err_range_now;
                BigDecimal b = new BigDecimal(new_value);
                this.td_dirty[i] = b.setScale(8, RoundingMode.HALF_UP).doubleValue();
            } else {
                this.td_dirty[i] = td_clean[i];
            }
        }
    }

    public double[] getTd_dirty() {
        return td_dirty;
    }

    public void writeRepairResultToFile(String targetFileName) {
        File writeFile = new File(targetFileName);
        try {
            BufferedWriter writeText = new BufferedWriter(new FileWriter(writeFile));
            writeText.write("timestamp,value");
            for (int j = 0; j < this.td_dirty.length; j++) {
                writeText.newLine();
                double val = this.td_dirty[j];
                writeText.write(j + "," + val);
            }
            writeText.flush();
            writeText.close();
        } catch (IOException e) {
            System.out.println("Error");
        }
    }

}
