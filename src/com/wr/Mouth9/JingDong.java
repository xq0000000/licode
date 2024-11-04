package com.wr.Mouth9;

import java.util.Arrays;
import java.util.Scanner;

/**
 * ClassName: JingDong
 * Description:
 * date: 2024/9/21 16:29
 *
 * @author Wang
 * @since JDK 1.8
 */
public class JingDong {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = 0, k = 0;
        int[] x = new int[n];
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
        }
    }
//    public static void main(String[] args) {
//        Scanner scanner = new Scanner(System.in);
//        int n = scanner.nextInt();
//        int k = scanner.nextInt();
//        int m=0,s = 0;
//        int sum = 1;
//        int[] a = new int[n];
//        int[] b = new int[2*n];
//        while (n-->0) {
//            a[m++] = scanner.nextInt();
//        }
//        for (int i = 1; i < a.length; i++) {
//            b[s++] = a[i-1];
//            b[s++] = a[i];
//            int max = max1(b);
//            int min = min1(b);
//            if (max - min > k) {
//                sum++;
//                Arrays.fill(b,-1);
//            }
//        }
//        System.out.println(sum);
//    }
//    public static void main(String[] args) {
//        Scanner scanner = new Scanner(System.in);
//        int n = scanner.nextInt();
//        int k = scanner.nextInt();
//        int m = 0;
//        int sum = 1;
//        int[] a = new int[n];
//        while (n-- > 0) {
//            a[m++] = scanner.nextInt();
//        }
//        for (int i = 1; i < a.length; i++) {
//            if (Math.abs(a[i] - a[i - 1]) > k) {
//                sum++;
//            }
//        }
//        System.out.println(sum);
//    }

    public static int max1(int[] a) {
        int max = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > max) {
                max = a[i];
            }
        }
        return max;
    }
    public static int min1(int[] a) {
        int min = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] < min&&a[i] > 0) {
                min = a[i];
            }
        }
        return min;
    }
}
