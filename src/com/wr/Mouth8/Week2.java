package com.wr.Mouth8;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * ClassName: Week2
 * Description:
 * date: 2024/8/18 10:31
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Week2 {
    /**
     * Q1. 统计满足 K 约束的子字符串数量 I
     * 给你一个 二进制 字符串 s 和一个整数 k。
     * 如果一个 二进制字符串 满足以下任一条件，则认为该字符串满足
     * k 约束：字符串中 0 的数量最多为 k。
     *     字符串中 1 的数量最多为 k。
     * 返回一个整数，表示 s 的所有满足 k 约束 的子字符串的数量。
     * @param s
     * @param k
     * @return
     */
    public static int countKConstraintSubstrings(String s, int k) {
        ArrayList<String> substrings = new ArrayList<>();
        int m = 0;
        int sum = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) {
                substrings.add(s.substring(i,j));
            }
        }
        for (String substring : substrings) {
            int p = 0;
            int q = 0;
            for (int j = 0; j < substring.length(); j++) {
                if (substring.charAt(j) == '1') {
                    p++;
                }
                if (substring.charAt(j) == '0') {
                    q++;
                }
            }
            if (p<=k || q<=k) {
                sum ++;
            }
        }
        return sum;
    }

    /**
     * Q2. 超级饮料的最大强化能量
     * 来自未来的体育科学家给你两个整数数组 energyDrinkA 和
     * energyDrinkB，数组长度都等于 n。这两个数组分别代表 A、B
     * 两种不同能量饮料每小时所能提供的强化能量。
     * 你需要每小时饮用一种能量饮料来 最大化 你的总强化能量。然而，
     * 如果从一种能量饮料切换到另一种，你需要等待一小时来梳理身体的
     * 能量体系（在那个小时里你将不会获得任何强化能量）。
     * 返回在接下来的 n 小时内你能获得的 最大 总强化能量。
     * 注意 你可以选择从饮用任意一种能量饮料开始。
     * @param energyDrinkA
     * @param energyDrinkB
     * @return
     */
    public static long maxEnergyBoost(int[] energyDrinkA, int[] energyDrinkB) {
        int n = energyDrinkA.length;
        long[][] dp = new long[n+1][2];
        for (int i = n - 1; i >= 0; i--) {
            //如果从一种能量饮料切换到另一种，你需要等待一小时来梳理身体的
            //如果一开始喝的A：1.然后又喝B,就会少一次能量。2.然后接着喝A,不会少能量，需要加上所获得的能量。
            dp[i][0] = Math.max(dp[i + 1][1], dp[i + 1][0] + energyDrinkA[i]);
            //如果一开始喝的B：1.然后又喝A,就会少一次能量。2.然后接着喝B,不会少能量，需要加上所获得的能量。
            dp[i][1] = Math.max(dp[i + 1][0], dp[i + 1][1] + energyDrinkB[i]);
        }
        return Math.max(dp[0][0], dp[0][1]);
    }

    /**
     * Q3. 找出最大的 N 位 K 回文数
     * 给你两个 正整数 n 和 k。
     * 如果整数 x 满足以下全部条件，则该整数是一个 k 回文数：
     *     x 是一个回文数。x 可以被 k 整除。
     * 以字符串形式返回 最大的  n 位 k 回文数。
     * 注意，该整数 不 含前导零。
     * @param n
     * @param k
     * @return
     */
    public static String largestPalindrome(int n, int k) {
            // 判断是否有效
            if (n <= 0 || k <= 0) return "";

            // 生成最大的回文数
            long maxNum = (long) Math.pow(10, n) - 1;

            // 从最大回文数开始搜索
            for (long i = maxNum; i >= 0; i--) {
                // 生成当前数的回文形式
                String numStr = String.valueOf(i);
                if (isPalindrome(numStr) && i % k == 0) {
                    return numStr;
                }
            }

            return "";
        }

        private static boolean isPalindrome(String s) {
            return s.equals(new StringBuilder(s).reverse().toString());
        }

    public static void main(String[] args) {
        int n = 5, k = 6;
        int[] energyDrinkA = {1,3,1}, energyDrinkB = {3,1,1};
        System.out.println(maxEnergyBoost(energyDrinkA, energyDrinkB));
    }
}
