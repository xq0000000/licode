package com.wr.Mouth8;

import com.wr.Day5_30.TreeNode;

import java.util.ArrayList;
import java.util.List;

/**
 * ClassName: Week1
 * Description: 第 410 场周赛
 * date: 2024/8/11 10:16
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Week1 {

    /**
     * Q1. 矩阵中的蛇
     * 大小为 n x n 的矩阵 grid 中有一条蛇。蛇可以朝 四个可能的方向 移动。
     * 矩阵中的每个单元格都使用位置进行标识： grid[i][j] = (i * n) + j。
     * 蛇从单元格 0 开始，并遵循一系列命令移动。
     * 给你一个整数 n 表示 grid 的大小，另给你一个字符串数组 commands，
     * 其中包括 "UP"、"RIGHT"、"DOWN" 和 "LEFT"。
     * 题目测评数据保证蛇在整个移动过程中将始终位于 grid 边界内。
     * 返回执行 commands 后蛇所停留的最终单元格的位置。
     * @param n
     * @param commands
     * @return
     */
    public static int finalPositionOfSnake(int n, List<String> commands) {
        int[][] grid= new int[n][n];
        final int[] i = {0};
        final int[] j = {0};
        commands.forEach(c->{
            if ("RIGHT".equals(c)) {
                j[0]++;
            }
            if ("DOWN".equals(c)) {
                i[0]++;
            }
            if ("LEFT".equals(c)) {
                if(j[0] >0) {
                    j[0]--;
                }
            }
            if ("UP".equals(c)) {
                if(i[0] >0) {
                    i[0]--;
                }
            }
        });
        int a = (i[0] * n) + j[0];
        return a;
    }

    /**
     * Q2. 统计好节点的数目
     * 现有一棵 无向 树，树中包含 n 个节点，按从 0 到 n - 1 标记。
     * 树的根节点是节点 0 。给你一个长度为 n - 1 的二维整数数组 edges，
     * 其中 edges[i] = [ai, bi] 表示树中节点 ai 与节点 bi 之间存在一条边。
     * 如果一个节点的所有子节点为根的子树包含的节点数相同，则认为该节点是一个好节点。
     * 返回给定树中 好节点 的数量。
     * 子树 指的是一个节点以及它所有后代节点构成的一棵树。
     * @param edges
     * @return
     */
    public static int countGoodNodes(int[][] edges) {
        TreeNode treeNode = new TreeNode();
        treeNode.val = 0;
        for (int i = 0; i < edges.length; i++) {

        }
        return 0;
    }

    /**
     * Q3. 单调数组对的数目 I
     * 给你一个长度为 n 的 正 整数数组 nums 。
     * 如果两个 非负 整数数组 (arr1, arr2) 满足以下条件，我们称它们是 单调 数组对：
     * 两个数组的长度都是 n 。
     * arr1 是单调 非递减 的，换句话说 arr1[0] <= arr1[1] <= ... <= arr1[n - 1] 。
     * arr2 是单调 非递增 的，换句话说 arr2[0] >= arr2[1] >= ... >= arr2[n - 1] 。
     * 对于所有的 0 <= i <= n - 1 都有 arr1[i] + arr2[i] == nums[i] 。
     * 请你返回所有 单调 数组对的数目。
     * 由于答案可能很大，请你将它对 109 + 7 取余 后返回。
     * @param nums
     * @return
     */
    public static int countOfPairs(int[] nums) {
        int[] a = new int[nums.length];
        int[] b = new int[nums.length];

        int sum = dfs(nums, a, b);
        return sum;
    }
    public static int dfs(int[] nums, int[] a, int[] b) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (a[i] + b[i] == nums[i] && a[i + 1] > a[i] && b[i + 1] < b[i]) {
                    k++;
                }
            }
        }
        return k;
    }

    public static void main(String[] args) {
        int[] nums = {2,3,2};
        int a = countOfPairs(nums);
        System.out.println(a);
    }
}
