import Algorithm.*;

import java.util.Arrays;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.File;
import java.util.Objects;

public class Experiment {
    //    private static final String dataPath = "./data/synthetic/ghi_10617120.csv";
//    private static final String dataPath = "./data/synthetic/power_5241600.csv";
//    private static final String dataPath = "./data/temp_grid.csv";

    public static Analysis srcdRepair(long[] td_time, double[] td_clean, double[] td_dirty, int period, double k, int max_iter, boolean[] td_bool) throws Exception {
        System.out.println("\nSRCD");
        SRCD srcd = new SRCD(td_time, td_dirty, period, k, max_iter);
        double[] td_repair = srcd.getTd_repair();
        long cost_time = srcd.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static Analysis srdRepair(long[] td_time, double[] td_clean, double[] td_dirty, int period, double k, int max_iter, boolean[] td_bool) throws Exception {
        System.out.println("\nSRRD");
        SRD srrd = new SRD(td_time, td_dirty, period, k, max_iter);
        double[] td_repair = srrd.getTd_repair();
        long cost_time = srrd.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static Analysis screenRepair(long[] td_time, double[] td_clean, double[] td_dirty, boolean[] td_bool) throws Exception {
        System.out.println("\nSCREEN");
        SCREEN screen = new SCREEN(td_time, td_dirty);
        double[] td_repair = screen.getTd_repair();
        long cost_time = screen.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static Analysis lsgreedyRepair(long[] td_time, double[] td_clean, double[] td_dirty, boolean[] td_bool) throws Exception {
        System.out.println("\nLsgreedy");
        Lsgreedy lsgreedy = new Lsgreedy(td_time, td_dirty);
        double[] td_repair = lsgreedy.getTd_repair();
        long cost_time = lsgreedy.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static Analysis imrRepair(long[] td_time, double[] td_clean, double[] td_dirty, double[] td_label, boolean[] td_bool) throws Exception {
        System.out.println("\nIMR");
        IMR imr = new IMR(td_time, td_dirty, td_label, td_bool);
        double[] td_repair = imr.getTd_repair();
        long cost_time = imr.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static Analysis ewmaRepair(long[] td_time, double[] td_clean, double[] td_dirty, boolean[] td_bool) throws Exception {
        System.out.println("\nEWMA");
        EWMA ewma = new EWMA(td_time, td_dirty);
        double[] td_repair = ewma.getTd_repair();
        long cost_time = ewma.getCost_time();
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time);
    }

    public static void recordRMSE(String string) throws Exception {
        FileWriter fileWritter = new FileWriter("./expRMSE.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void recordTime(String string) throws Exception {
        FileWriter fileWritter = new FileWriter("./expTime.txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void main(String[] args) throws Exception { //synthetic
        // parameter
        int dataLen = 22825440;
        // error
        int error_rate = 2; //error_rate/1000
        double error_range = 1.0;
        int error_length = 5;
        // 4seasonal
        int period = 1440;
        int max_iter = 10;
        double classical_k = 3.0;
        double robust_k = 15.0;
        // 4imr
        double label_rate = 0.5;
        // random
        int seed = 666;

        for (int dataset = 0; dataset < 2; dataset++) {
            dataLen = 22825440;
            int dataLenBase = dataset == 0 ? 1000000 : 4000000;
            String dataPath = dataset == 0 ? "./data/synthetic/power_5241600.csv" : "./data/synthetic/voltage_22825440.csv";
            period = dataset == 0 ? 144 : 1440;

            String record = dataPath.split("/")[3].split("_")[0] + " error_rate=" + error_rate + " error_range=" + error_range + " error_length=2 4 6 8 10" + " dataLenBase=" + dataLenBase + "\n";
            System.out.print(record);
            recordRMSE(record);
            recordTime(record);
            for (error_length = 2; error_length <= 10; error_length += 2) {
                // start
                LoadData loadData = new LoadData(dataPath, dataLen);
                long[] td_time = loadData.getTd_time();
                double[] td_clean = loadData.getTd_clean();
                System.out.println("finish load data");

                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, seed);
                double[] td_dirty_complete = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty_complete, label_rate, seed);
                double[] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                Analysis analysis;
                double[] td_dirty;
                dataLen = dataLenBase * 2;
                td_dirty = new double[dataLen];
                System.arraycopy(td_dirty_complete, 0, td_dirty, 0, td_dirty.length);
                for (int j = 1; j < 2; j++) {
                    switch (j) {
                        case 0 -> analysis = srcdRepair(td_time, td_clean, td_dirty, period, classical_k, max_iter, default_bool);
                        case 1 -> analysis = srdRepair(td_time, td_clean, td_dirty, period, robust_k, max_iter, default_bool);
                        case 2 -> analysis = screenRepair(td_time, td_clean, td_dirty, default_bool);
                        case 3 -> analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool);
                        case 4 -> analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool);
                        default -> analysis = ewmaRepair(td_time, td_clean, td_dirty, default_bool);
                    }
                    recordRMSE(analysis.getRMSE() + ",");
                    recordTime(analysis.getCost_time() + ",");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordRMSE("\n");
                recordTime("\n");
            }
            recordRMSE("\n");
            recordTime("\n");
        }
        // repair
//        Analysis srcd = srcdRepair(td_time, td_clean, td_dirty, period, classical_k, max_iter, default_bool);
//        Analysis srrd = srrdRepair(td_time, td_clean, td_dirty, period, robust_k, max_iter, default_bool);
//        Analysis screen = screenRepair(td_time, td_clean, td_dirty, default_bool);
//        Analysis lsgreedy = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool);
//        Analysis imr = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool);
//        Analysis ewma = ewmaRepair(td_time, td_clean, td_dirty, default_bool);

        //save
//        String saveName =  dataPath.split("/")[2];
//        srcd.writeRepairResultToFile("./output/srcd_" + saveName + "_repair.csv");
//        srrd.writeRepairResultToFile("./output/srrd_" + saveName + "_repair.csv");
//        screen.writeRepairResultToFile("./output/screen_" + saveName + "repair.csv");
//        lsgreedy.writeRepairResultToFile("./output/lsgreedy_" + saveName + "_repair.csv");
//        imr.writeRepairResultToFile("./output/imr_" + saveName + "_repair.csv");
//        ewma.writeRepairResultToFile("./output/ewma_" + saveName + "_repair.csv");

//        System.out.println(
//                "RMSE:\n" +
//                        "srcd : " + srcd.getRMSE() + "\n" +
//                        "srrd : " + srrd.getRMSE() + "\n" +
//                        "screen : " + screen.getRMSE() + "\n" +
//                        "lsgreedy : " + lsgreedy.getRMSE() + "\n" +
//                        "imr : " + imr.getRMSE() + "\n" +
//                        "ewma : " + ewma.getRMSE() + "\n" +
//                        ""
//        );
//
//        System.out.println(
//                "Cost Time:\n" +
//                        "srcd : " + srcd.getCost_time() + "\n" +
//                        "srrd : " + srrd.getCost_time() + "\n" +
//                        "screen : " + screen.getCost_time() + "\n" +
//                        "lsgreedy : " + lsgreedy.getCost_time() + "\n" +
//                        "imr : " + imr.getCost_time() + "\n" +
//                        "ewma : " + ewma.getCost_time() + "\n" +
//                        ""
//        );
    }

}
