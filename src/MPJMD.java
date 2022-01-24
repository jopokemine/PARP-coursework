
/*
 * "Physics" part of code adapted from Dan Schroeder's applet at:
 *
 *     http://physics.weber.edu/schroeder/software/MDapplet.html
 */

import java.awt.*;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.ReentrantLock;

import mpi.*;

import javax.swing.*;

public class MPJMD {

    // Size of simulation

    final static int N = 4000; // Number of "atoms"
    final static double BOX_WIDTH = 100.0;

    // Benchmarking
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
    final static int OUTPUT_FREQ = 2500;

    // Physics constants

    final static double ATOM_RADIUS = 0.5;

    final static double WALL_STIFFNESS = 500.0;
    final static double GRAVITY = 0.005;
    final static double FORCE_CUTOFF = 3.0;

    // Atom positions
    static double[] x = new double[N];
    static double[] y = new double[N];

    // Atom velocities
    static double[] vx = new double[N];
    static double[] vy = new double[N];

    // Atom accelerations
    static double[] ax = new double[N];
    static double[] ay = new double[N];

    static Display display = new Display();

    public static void main(String args[]) throws Exception {

        MPI.Init(args);

        int me = MPI.COMM_WORLD.Rank();
        int P = MPI.COMM_WORLD.Size();

        // Define initial state of atoms

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
        Thread.sleep(DELAY);

        long start = System.currentTimeMillis();

        long end = System.currentTimeMillis();

        System.out.println("Completed in: " + (end - start) + " milliseconds");
    }

    public void run() {

        for (int iter = 0; iter < ITERATIONS; iter++) {

            if (iter % OUTPUT_FREQ == 0 && me == 0) {
                System.out.println("iter = " + iter + ", time = " + iter * DT);
                // Thread.sleep(DELAY);
            }

            // Verlet integration:
            // http://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet

            double dtOver2 = 0.5 * DT;
            double dtSquaredOver2 = 0.5 * DT * DT;

            for (int i = begin; i < end; i++) {
                x[i] += (vx[i] * DT) + (ax[i] * dtSquaredOver2);
                // update position
                y[i] += (vy[i] * DT) + (ay[i] * dtSquaredOver2);
                vx[i] += (ax[i] * dtOver2); // update velocity halfway
                vy[i] += (ay[i] * dtOver2);
            }

            computeAccelerations();

            synch();

            for (int i = begin; i < end; i++) {
                vx[i] += (ax[i] * dtOver2);
                // finish updating velocity with new acceleration
                vy[i] += (ay[i] * dtOver2);
            }

            synch();

            if (iter % OUTPUT_FREQ == 0 && me == 0) {
                display.repaint();
            }

        }
    }

    // Compute accelerations of all atoms from current positions:
    void computeAccelerations() {

        double dx, dy; // separations in x and y directions
        double dx2, dy2, rSquared, rSquaredInv, attract, repel, fOverR, fx, fy;

        // first check for bounces off walls, and include gravity (if any):
        if (me == 0) {
            for (int i = 0; i < N; i++) {
                if (x[i] < ATOM_RADIUS) {
                    ax[i] = WALL_STIFFNESS * (ATOM_RADIUS - x[i]);
                } else if (x[i] > (BOX_WIDTH - ATOM_RADIUS)) {
                    ax[i] = WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - x[i]);
                } else {
                    ax[i] = 0.0;
                }
                if (y[i] < ATOM_RADIUS) {
                    ay[i] = (WALL_STIFFNESS * (ATOM_RADIUS - y[i]));
                } else if (y[i] > (BOX_WIDTH - ATOM_RADIUS)) {
                    ay[i] = (WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - y[i]));
                } else {
                    ay[i] = 0;
                }
                ay[i] -= GRAVITY;
            }
        }

        double forceCutoff2 = FORCE_CUTOFF * FORCE_CUTOFF;

        synch();

        // Now compute interaction forces (Lennard-Jones potential).
        // This is where the program spends most of its time.

        // (NOTE: use of Newton's 3rd law below to essentially half number
        // of calculations needs some care in a parallel version.
        // A naive decomposition on the i loop can lead to a race condition
        // because you are assigning to ax[j], etc.
        // You can remove these assignments and extend the j loop to a fixed
        // upper bound of N, or, for extra credit, find a cleverer solution!)

        for (int i = begin + 1; i < end; i++) {
            for (int j = 0; j < i; j++) { // loop over all distinct pairs
                dx = x[i] - x[j];
                dx2 = dx * dx;
                if (dx2 < forceCutoff2) { // make sure they're close enough to bother
                    dy = y[i] - y[j];
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

                            // Critical section - implemented locks to prevent race condition
                            lock.lock();
                            try {
                                ax[i] += fx; // add this force on to i's acceleration (mass = 1)
                                ay[i] += fy;

                                ax[j] -= fx; // Newton's 3rd law
                                ay[j] -= fy;
                            } finally {
                                lock.unlock();
                            }
                        }
                    }
                }
            }
        }
    }

    static void synch() {
        try {
            barrier.await();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    static class Display extends JPanel {

        static final double SCALE = WINDOW_SIZE / BOX_WIDTH;

        static final int DIAMETER = Math.max((int) (SCALE * 2 * ATOM_RADIUS), 2);

        Display() {

            setPreferredSize(new Dimension(WINDOW_SIZE, WINDOW_SIZE));

            JFrame frame = new JFrame("MPJMD");
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
