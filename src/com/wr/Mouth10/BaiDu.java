package com.wr.Mouth10;

import java.util.Scanner;

/**
 * ClassName: BaiDu
 * Description:
 * date: 2024/10/15 19:25
 *
 * @author Wang
 * @since JDK 1.8
 */
public class BaiDu {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int T = scanner.nextInt();
        int[] n1 = new int[T];
        int[] k1 = new int[T];
        int m = 0;
        while (m+1 <= T) {
            int n = scanner.nextInt();
            int k = scanner.nextInt();
            n1[m] = n;
            if( n> k) {
                k1[m] = k;
            } else {
                k1[m] = 1;
            }
            m++;
        }
        for (int i : k1) {
            System.out.println(i);
        }
    }
}
