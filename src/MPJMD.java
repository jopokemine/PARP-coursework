
/*
 * "Physics" part of code adapted from Dan Schroeder's applet at:
 *
 *     http://physics.weber.edu/schroeder/software/MDapplet.html
 */

import java.awt.*;
import java.util.Arrays;

import mpi.*;

import javax.swing.*;

public class MPJMD {

    // Size of simulation

    final static int N = 4000; // Number of "atoms"
    final static double BOX_WIDTH = 100.0;

    // Iterations for equal benchmarking
    final static int ITERATIONS = 5000;

    // Initial state - controls temperature of system

    // final static double VELOCITY = 3.0 ; // gaseous
    final static double VELOCITY = 2.0; // gaseous/"liquid"
    // final static double VELOCITY = 1.0; // "crystalline"

    final static double INIT_SEPARATION = 2.2; // in atomic radii

    // Simulation

    final static double DT = 0.01; // Time step

    // Display

    final static int WINDOW_SIZE = 800;
    final static int DELAY = 0;
    final static int OUTPUT_FREQ = 1;

    // Physics constants

    final static double ATOM_RADIUS = 0.5;

    final static double WALL_STIFFNESS = 500.0;
    final static double GRAVITY = 0.005;
    final static double FORCE_CUTOFF = 3.0;

    final static int BLOCK_SIZE = 8;

    final static int BUFFER_SIZE = 1 + 2 * N;
    // final static int RESULT_SIZE = 1 + 2 * BLOCK_SIZE;
    final static int NUM_BLOCKS = N / BLOCK_SIZE;

    final static int TAG_HELLO = 0;
    final static int TAG_TASK = 1;
    final static int TAG_RESULT = 2;
    final static int TAG_GOODBYE = 3;

    public static void main(String args[]) throws Exception {

        MPI.Init(args);

        int me = MPI.COMM_WORLD.Rank();
        int P = MPI.COMM_WORLD.Size();

        int numWorkers = P - 1;

        // Atom positions
        double[] x = new double[N];
        double[] y = new double[N];
        double[] buffer = new double[BUFFER_SIZE];
        double[] resultBuffer = new double[BUFFER_SIZE];
        /*
         * buffer[0] == blockStart
         * buffer[i <= N] == x
         * buffer[i > N] == y
         */

        // Atom velocities
        double[] vx = new double[N];
        double[] vy = new double[N];

        // Atom accelerations
        double[] ax = new double[N];
        double[] ay = new double[N];

        if (me == 0) {
            Display display = new Display();

            int sqrtN = (int) (Math.sqrt((double) N) + 0.5);
            double initSeparation = INIT_SEPARATION * ATOM_RADIUS;
            for (int i = 0; i < N; i++) {
                // lay out atoms regularly, so no overlap
                x[i] = (0.5 + i % sqrtN) * initSeparation;
                y[i] = (0.5 + i / sqrtN) * initSeparation;
                vx[i] = (2 * Math.random() - 1) * VELOCITY;
                vy[i] = (2 * Math.random() - 1) * VELOCITY;
            }

            display.repaint();

            long startTime = System.currentTimeMillis();

            int nextBlockStart;
            int numHellos;
            int numBlocksReceived;

            for (int iter = 0; iter < ITERATIONS; iter++) {

                if (iter % OUTPUT_FREQ == 0 && me == 0) {
                    System.out.println("iter = " + iter + ", time = " + iter * DT);
                }

                nextBlockStart = 0;
                numHellos = 0;
                numBlocksReceived = 0;

                double dtOver2 = 0.5 * DT;
                double dtSquaredOver2 = 0.5 * DT * DT;
                for (int i = 0; i < N; i++) {
                    x[i] += (vx[i] * DT) + (ax[i] * dtSquaredOver2);
                    // update position
                    y[i] += (vy[i] * DT) + (ay[i] * dtSquaredOver2);
                    vx[i] += (ax[i] * dtOver2); // update velocity halfway
                    vy[i] += (ay[i] * dtOver2);
                }

                while (numBlocksReceived < NUM_BLOCKS || numHellos < numWorkers) {

                    Status status = MPI.COMM_WORLD.Recv(resultBuffer, 0, BLOCK_SIZE, MPI.INT, MPI.ANY_SOURCE,
                            MPI.ANY_TAG);

                    if (status.tag == TAG_RESULT) {
                        int resultBlockStart = (int) resultBuffer[0];
                        for (int i = 0; i < BUFFER_SIZE; i++) {
                            if (i < N) {
                                ax[i] += resultBuffer[1 + i];
                            } else {
                                ay[i - N] += resultBuffer[1 + i];
                            }
                        }
                        numBlocksReceived++;
                    } else {
                        numHellos++;
                    }

                    if (nextBlockStart < N) {
                        buffer[0] = nextBlockStart;
                        for (int i = 1; i < BUFFER_SIZE; i++) {
                            if (i < N) { // x
                                buffer[i] = x[i - 1];
                            } else { // y
                                buffer[i] = y[i - 1 - N];
                            }
                        }
                        MPI.COMM_WORLD.Send(buffer, 0, BUFFER_SIZE, MPI.DOUBLE, status.source, TAG_TASK);
                        nextBlockStart += BLOCK_SIZE;
                        System.out.println("Sending work to " + status.source);
                    } else {
                        MPI.COMM_WORLD.Send(buffer, 0, 0, MPI.INT, status.source, TAG_GOODBYE);
                        System.out.println("Shutting down " + status.source);
                    }
                }

                for (int i = 0; i < N; i++) {
                    vx[i] += (ax[i] * dtOver2);
                    // finish updating velocity with new acceleration
                    vy[i] += (ay[i] * dtOver2);
                }

                if (iter % OUTPUT_FREQ == 0) {
                    display.repaint();
                }

            }

            long endTime = System.currentTimeMillis();

            System.out.println("Calculation completed in " +
                    (endTime - startTime) + " milliseconds");
        } else {
            MPI.COMM_WORLD.Send(resultBuffer, 0, 0, MPI.DOUBLE, 0, TAG_HELLO);

            boolean done = false;

            while (!done) {
                Status status = MPI.COMM_WORLD.Recv(buffer, 0, BUFFER_SIZE, MPI.DOUBLE, 0, ANY.TAG);

                if (status.tag == TAG_TASK) {
                    int blockStart = (int) buffer[0];
                    int blockEnd = blockStart + BLOCK_SIZE;
                    double[] blockX = Arrays.copyOfRange(buffer, 1, N);
                    double[] blockY = Arrays.copyOfRange(buffer, N, buffer.length);
                    resultBuffer[0] = blockStart;

                    double dx, dy; // separations in x and y directions
                    double dx2, dy2, rSquared, rSquaredInv, attract, repel, fOverR, fx, fy;

                    // first check for bounces off walls, and include gravity (if any):
                    for (int i = 0; i < blockEnd; i++) {
                        if (blockX[i] < ATOM_RADIUS) {
                            resultBuffer[i + 1] = WALL_STIFFNESS * (ATOM_RADIUS - blockX[i]);
                        } else if (blockX[blockStart + i] > (BOX_WIDTH - ATOM_RADIUS)) {
                            resultBuffer[i + 1] = WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - blockX[i]);
                        } else {
                            resultBuffer[i + 1] = 0.0;
                        }
                        if (blockY[i] < ATOM_RADIUS) {
                            resultBuffer[i + 1 + BLOCK_SIZE] = (WALL_STIFFNESS * (ATOM_RADIUS - blockY[i]));
                        } else if (blockY[i] > (BOX_WIDTH - ATOM_RADIUS)) {
                            resultBuffer[i + 1 + BLOCK_SIZE] = (WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - blockY[i]));
                        } else {
                            resultBuffer[i + 1 + BLOCK_SIZE] = 0;
                        }
                        resultBuffer[i + 1 + BLOCK_SIZE] -= GRAVITY;
                    }

                    double forceCutoff2 = FORCE_CUTOFF * FORCE_CUTOFF;

                    // Now compute interaction forces (Lennard-Jones potential).
                    // This is where the program spends most of its time.

                    // (NOTE: use of Newton's 3rd law below to essentially half number
                    // of calculations needs some care in a parallel version.
                    // A naive decomposition on the i loop can lead to a race condition
                    // because you are assigning to ax[j], etc.
                    // You can remove these assignments and extend the j loop to a fixed
                    // upper bound of N, or, for extra credit, find a cleverer solution!)

                    for (int i = 1; i < blockEnd; i++) {
                        for (int j = 0; j < i; j++) { // loop over all distinct pairs
                            dx = blockX[i] - blockX[j];
                            dx2 = dx * dx;
                            if (dx2 < forceCutoff2) { // make sure they're close enough to bother
                                dy = blockY[i] - blockY[j];
                                dy2 = dy * dy;
                                if (dy2 < forceCutoff2) {
                                    rSquared = dx2 + dy2;
                                    if (rSquared < forceCutoff2) {
                                        rSquaredInv = 1.0 / rSquared;
                                        attract = rSquaredInv * rSquaredInv * rSquaredInv;
                                        repel = attract * attract;
                                        fOverR = 24.0 * ((2.0 * repel) - attract) * rSquaredInv;
                                        fx = fOverR * dx;
                                        fy = fOverR * dy;
                                        resultBuffer[i] += fx; // add this force on to i's acceleration (mass = 1)
                                        resultBuffer[i + N] += fy;

                                        resultBuffer[j + 1] -= fx; // Newton's 3rd law
                                        resultBuffer[j + 1 + N] -= fy;
                                    }
                                }
                            }
                        }
                    }

                    MPI.COMM_WORLD.Send(resultBuffer, 0, RESULT_SIZE, MPI.DOUBLE, 0, TAG_RESULT);
                } else { // TAG_GOODBYE
                    done = true;
                }
            }
        }

        MPI.Finalize();
    }

    static class Display extends JPanel {

        static final double SCALE = WINDOW_SIZE / BOX_WIDTH;

        static final int DIAMETER = Math.max((int) (SCALE * 2 * ATOM_RADIUS), 2);

        Display() {

            setPreferredSize(new Dimension(WINDOW_SIZE, WINDOW_SIZE));

            JFrame frame = new JFrame("MD");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setContentPane(this);
            frame.pack();
            frame.setVisible(true);
        }

        public void paintComponent(Graphics g) {
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, WINDOW_SIZE, WINDOW_SIZE);
            g.setColor(Color.BLUE);
            for (int i = 0; i < N; i++) {
                g.fillOval((int) (SCALE * (x[i] - ATOM_RADIUS)),
                        WINDOW_SIZE - 1 - (int) (SCALE * (y[i] + ATOM_RADIUS)),
                        DIAMETER, DIAMETER);
            }
        }
    }
}
