package com.wr.Mouth8;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * ClassName: Week3
 * Description:
 * date: 2024/9/1 10:29
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Week3 {
    /**
     * Q1. 检查棋盘方格颜色是否相同
     * 给你两个字符串coordinate1和coordinate2，代表 8 x 8 国际象棋棋盘上
     * 的两个方格的坐标。以下是棋盘的参考图。
     * 如果这两个方格颜色相同，返回 true，否则返回 false。
     * 坐标总是表示有效的棋盘方格.坐标的格式总是先字母（表示列）,再数字（表示行）。
     * @param coordinate1
     * @param coordinate2
     * @return
     */
    public static boolean checkTwoChessboards(String coordinate1, String coordinate2) {
        int a = 0, b = 0;
        if((coordinate1.charAt(0) - 97 + 1) % 2 == coordinate1.charAt(1) % 2) {
            a = 1;
        }
        if((coordinate1.charAt(0) - 97 + 1) % 2 != coordinate1.charAt(1) % 2) {
            a = 0;
        }
        if((coordinate2.charAt(0) - 97 + 1) % 2 == coordinate2.charAt(1) % 2) {
            b = 1;
        }
        if((coordinate2.charAt(0) - 97 + 1) % 2 != coordinate2.charAt(1) % 2) {
            b = 0;
        }
        return a == b;
    }

    /**
     * Q2. 第 K 近障碍物查询
     * 有一个无限大的二维平面。
     * 给你一个正整数 k ，同时给你一个二维数组 queries ，包含一系列查询：
     * queries[i] = [x, y] ：在平面上坐标 (x, y) 处建一个障碍物，
     * 数据保证之前的查询不会在这个坐标处建立任何障碍物。
     * 每次查询后，你需要找到离原点第k近障碍物到原点的 距离 。
     * 请你返回一个整数数组 results ，其中 results[i] 表示建立第 i 个障碍物以后
     * 离原地第 k 近障碍物距离原点的距离。如果少于 k 个障碍物，results[i] == -1 。
     * 注意，一开始没有任何障碍物。
     * 坐标在 (x, y) 处的点距离原点的距离定义为 |x| + |y| 。
     * @param queries
     * @param k
     * @return
     */
    public static int[] resultsArray(int[][] queries, int k) {
        int n = queries.length;
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            if(i<k-1) {
                result[i] = -1;
            } else {
                int[] a = new int[i+1];
                for (int j = 0; j <= i; j++) {
                    a[j] = Math.abs(queries[j][0]) + Math.abs(queries[j][1]);
                }
                Arrays.sort(a);
                result[i] = a[k-1];
            }
        }
        return result;
    }
    public static int[] getKClosestDistances(int[][] queries, int k) {
        int n = queries.length;
        int[] result = new int[n];
        // 使用最大堆来维护前 k 个最近的障碍物的距离
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

        for (int i = 0; i < n; i++) {
            int x = queries[i][0];
            int y = queries[i][1];
            int distance = Math.abs(x) + Math.abs(y);

            // 添加当前距离到堆
            maxHeap.offer(distance);

            // 如果堆的大小超过 k，移除堆顶元素
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }

            // 如果堆的大小达到 k，堆顶元素就是第 k 近的距离
            if (maxHeap.size() == k) {
                result[i] = maxHeap.peek();
            } else {
                result[i] = -1;
            }
        }

        return result;
    }



    /**
     * Q3. 选择矩阵中单元格的最大得分
     * 给你一个由正整数构成的二维矩阵 grid。
     * 你需要从矩阵中选择 一个或多个 单元格，选中的单元格应满足以下条件：
     * 所选单元格中的任意两个单元格都不会处于矩阵的 同一行。
     * 所选单元格的值 互不相同。
     * 你的得分为所选单元格值的总和。
     * 返回你能获得的 最大 得分。
     * @param grid
     * @return
     */
    public int maxScore(List<List<Integer>> grid) {

        return 0;
    }

    public static void main(String[] args) {
        int[][] queries = {{1,2},{3,4},{2,3},{-3,0}};
        int k = 2;
        System.out.println(Arrays.toString(getKClosestDistances(queries, k)));
    }
}
