package com.wr.Mouth8;

import com.wr.base.TreeNode;

import java.util.*;
import java.util.stream.Collectors;

/**
 * ClassName: Solution
 * Description:
 * date: 2024/8/8 9:23
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Solution {

    /**
     * 3131. 找出与数组相加的整数 I
     * 给你两个长度相等的数组 nums1 和 nums2。
     * 数组 nums1 中的每个元素都与变量 x 所表示的整数相加。
     * 如果 x 为负数，则表现为元素值的减少。
     * 在与 x 相加后，nums1 和 nums2 相等 。当两个数组中包含相同的整数，
     * 并且这些整数出现的频次相同时，两个数组相等 。返回整数 x 。
     * @param nums1
     * @param nums2
     * @return
     */
    public static int addedInteger(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        return nums2[0]-nums1[0];
    }

    /**
     * 70. 爬楼梯
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * @param n
     * @return
     */
    public static int climbStairs(int n) {
        int[] memo = new int[n+1];
        return dfs(n, memo);
    }
    private static int dfs(int n, int[] memo) {
        if ( n <= 1 ){
            return 1;
        }
        if (memo[n] != 0) {
            return memo[n];
        }
        return memo[n] = dfs(n-1, memo) + dfs(n-2, memo);
    }

    /**
     * 1. 两数之和
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出
     * 和为目标值 target 的那两个整数，并返回它们的数组下标。
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * 你可以按任意顺序返回答案。
     * @param nums
     * @param target
     * @return
     */
    public static int[] twoSum(int[] nums, int target) {
        int[] a = new int[2];
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    a[0] = i;
                    a[1] = j;
                }
            }
        }
        return a;

    }

    /**
     * 88. 合并两个有序数组
     * 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，
     * 分别表示 nums1 和 nums2 中的元素数目。
     * 请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
     * 注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，
     * nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，
     * 应忽略。nums2 的长度为 n 。
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public static void merge1(int[] nums1, int m, int[] nums2, int n) {
        for (int i = 0; i < n; i++) {
            nums1[m+i] = nums2[i];
        }
        Arrays.sort(nums1);
    }

    public void merge2(int[] nums1, int m, int[] nums2, int n) {
        int p1 = 0, p2 = 0;
        int[] sorted = new int[m + n];
        int cur;
        while (p1 < m || p2 < n) {
            if (p1 == m) {
                cur = nums2[p2++];
            } else if (p2 == n) {
                cur = nums1[p1++];
            } else if (nums1[p1] < nums2[p2]) {
                cur = nums1[p1++];
            } else {
                cur = nums2[p2++];
            }
            sorted[p1 + p2 - 1] = cur;
        }
        for (int i = 0; i != m + n; ++i) {
            nums1[i] = sorted[i];
        }
    }

    /**
     * 392. 判断子序列
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     * 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。
     * （例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     * 如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，
     * 你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
     * @param s
     * @param t
     * @return
     */
    public static boolean isSubsequence(String s, String t) {
        if ("".equals(s)) { //若为空则一定为子串
            return true;
        }
        int i = 0;
        for (int j = 0; j < t.length(); j++) { // 同向双指针
            if (t.charAt(j) == s.charAt(i) && ++i == s.length()) {
                return true;
            }
        }
        return false;
    }

    /**
     * 3132. 找出与数组相加的整数 II
     * 给你两个整数数组 nums1 和 nums2。
     * 从 nums1 中移除两个元素，并且所有其他元素都与变量 x 所表示的整数相加。
     * 如果 x 为负数，则表现为元素值的减少。
     * 执行上述操作后，nums1 和 nums2 相等 。当两个数组中包含相同的整数，
     * 并且这些整数出现的频次相同时，两个数组 相等 。
     * 返回能够实现数组相等的 最小 整数 x 。
     * @param nums1
     * @param nums2
     * @return
     */
    public static int minimumAddedInteger(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        for (int i = 2; i >= 0; i--) {
            int x = nums2[0]-nums1[i];
            int j = 0;
            for (int k = 0; k < nums1.length; k++) {
                if (nums1[k]+x == nums2[j] && ++j == nums2.length) {
                    return x;
                }
            }
        }
        return 0;
    }

    /**
     * 509. 斐波那契数
     * 斐波那契数 （通常用 F(n) 表示）形成的序列称为 斐波那契数列 。
     * 该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
     * F(0) = 0，F(1) = 1
     * F(n) = F(n - 1) + F(n - 2)，其中 n > 1
     * 给定 n ，请计算 F(n) 。
     * @param n
     * @return
     */
    public static int fib(int n) {
        int[] memo = new int[n+1];
        return dfs1(n, memo);
    }
    public static int dfs1(int n, int[] memo) {
        if (n == 0) {
            return memo[n] = 0;
        }
        if (n == 1) {
            return memo[n] = 1;
        }
        if (memo[n] != 0) {
            return memo[n];
        }
        return memo[n] = dfs1(n-1 , memo) + dfs1(n-2, memo);
    }

    /**
     * 1137. 第 N 个泰波那契数
     * 泰波那契序列 Tn 定义如下：
     * T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下
     * Tn+3 = Tn + Tn+1 + Tn+2
     * 给你整数 n，请返回第 n 个泰波那契数 Tn 的值。
     * @param n
     * @return
     */
    public static int tribonacci(int n) {
        int[] memo = new int[n+1];
        return dfs2(n, memo);
    }
    public static int dfs2(int n, int[] memo) {
        if (memo[n] != 0) {
            return memo[n];
        }
        if (n == 0) {
            return memo[n] = 0;
        }
        if (n == 1) {
            return memo[n] = 1;
        }
        if (n == 2) {
            return memo[n] = 1;
        }
        return memo[n] = dfs2(n-1 , memo) + dfs2(n-2, memo) +dfs2(n-3, memo);
    }

    /**
     * 27. 移除元素
     * 给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素。
     * 元素的顺序可能发生改变。然后返回nums中与val不同的元素的数量。
     * 假设 nums 中不等于 val 的元素数量为 k，要通过此题，您需要执行以下操作：
     * 更改 nums 数组，使 nums 的前 k 个元素包含不等于 val 的元素。
     * nums 的其余元素和 nums 的大小并不重要。返回 k。
     * @param nums
     * @param val
     * @return
     */
    public static int removeElement(int[] nums, int val) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val) {
                nums[k] = nums[i];
                k++;
            }
        }
        System.out.println(Arrays.toString(nums));
        return k;
    }

    /**
     * 676. 实现一个魔法字典
     * 设计一个使用单词列表进行初始化的数据结构，单词列表中的单词互不相同 。
     * 如果给出一个单词，请判定能否只将这个单词中一个字母换成另一个字母，
     * 使得所形成的新单词存在于你构建的字典中。
     * 实现 MagicDictionary 类：
     * MagicDictionary() 初始化对象
     * void buildDict(String[] dictionary) 使用字符串数组 dictionary 设定该数据结构，
     * dictionary 中的字符串互不相同 bool search(String searchWord)
     * 给定一个字符串 searchWord ，判定能否只将字符串中一个字母换成另一个字母，使得所形成的新字符
     * 串能够与字典中的任一字符串匹配。如果可以，返回 true ；否则，返回 false 。
     * Your MagicDictionary object will be instantiated and called as such:
     * MagicDictionary obj = new MagicDictionary();
     * obj.buildDict(dictionary);
     * boolean param_2 = obj.search(searchWord);
     */
    class MagicDictionary {
        private String[] words;

        public MagicDictionary() {

        }

        public void buildDict(String[] dictionary) {
            words = dictionary;
        }

        public boolean search(String searchWord) {
            for (String word : words) {
                if (searchWord.length() != word.length()) {
                    continue;
                }
                int diff = 0;
                for (int i = 0; i < word.length(); i++) {
                    if (word.charAt(i) != searchWord.charAt(i)) {
                        diff++;
                    }

                    if (diff > 1) {
                        break;
                    }
                }
                if (diff == 1) {
                    return true;
                }
            }
            return false;
        }

    }

    /**
     * 104. 二叉树的最大深度
     * 给定一个二叉树 root ，返回其最大深度。
     * 二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
     * @param root
     * @return
     */
    public static int maxDepth(TreeNode root) {
        if (root.val == 0) {
            return 0;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.left);
        return Math.max(left,right) + 1;
    }

    /**
     * 144. 二叉树的前序遍历
     * 给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        preorder(root, list);
        return list;
    }

    public void preorder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        preorder(root.left, list);
        preorder(root.right, list);
    }

    /**
     * 543. 二叉树的直径
     * 给你一棵二叉树的根节点，返回该树的 直径 。
     * 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。
     * 这条路径可能经过也可能不经过根节点 root 。
     * 两节点之间路径的 长度 由它们之间边数表示。
     * @param root
     * @return
     */

    int ans;
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
    public int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left =  depth(root.left);
        int right =  depth(root.right);
        ans = Math.max(ans , left+right+1);
        return Math.max(left, right) + 1;
    }

    /**
     * 3151. 特殊数组 I
     * 如果数组的每一对相邻元素都是两个奇偶性不同的数字，则该数组被认为是一个 特殊数组 。
     * Aging有一个整数数组nums。如果nums是一个特殊数组,返回true,否则返回 false。
     * @param nums
     * @return
     */
    public static boolean isArraySpecial(int[] nums) {
        if (nums.length == 1) {
            return true;
        }
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] %2==nums[i-1]%2) {
                return false;
            }
        }
        return true;
    }

    /**
     * 746. 使用最小花费爬楼梯
     * 给你一个整数数组 cost,其中cost[i]是从楼梯第i个台阶向上爬需要支付的费用。
     * 一旦你支付此费用，即可选择向上爬一个或者两个台阶。
     * 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
     * 请你计算并返回达到楼梯顶部的最低花费。
     * @param cost
     * @return
     */
    public static int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] memo = new int[n+1];
        Arrays.fill(memo, -1);
        return dfs(memo, n, cost);
    }
    private static int dfs(int[] memo, int i, int[] cost) {
        if (i <= 1) {
            return 0;
        }
        if (memo[i] != -1) {
            return memo[i];
        }
        int n1 = dfs(memo, i-1, cost) + cost[i-1];
        int n2 = dfs(memo, i-2, cost) + cost[i-2];
        return memo[i] = Math.min(n1,n2);
     }

    /**
     * 198. 打家劫舍
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，
     * 影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，
     * 一夜之内能够偷窃到的最高金额。
     * @param nums
     * @return
     */
    public static int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
        }
        return dp[n-1];
    }

    /**
     * 3152. 特殊数组 II
     * 如果数组的每一对相邻元素都是两个奇偶性不同的数字，则该数组被认为是一个特殊数组。
     * 周洋哥有一个整数数组nums和一个二维整数矩阵queries，对于
     * queries[i]=[fromi,toi],请你帮助周洋哥检查子数组nums[fromi..toi]
     * 是不是一个特殊数组。返回布尔数组answer，如果nums[fromi..toi]是特殊数组，
     * 则 answer[i]为true,否则,answer[i]为 false 。
     * @param nums
     * @param queries
     * @return
     */
    public static boolean[] isArraySpecial(int[] nums, int[][] queries) {
        boolean[] b = new boolean[queries.length];
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < n; i++) {
            if (((nums[i] ^ nums[i - 1]) & 1) != 0) { //判断两个相邻数是否为奇数
                dp[i] = dp[i-1] + 1;
            }
        }
        for (int i = 0; i < queries.length; i++) {
            int x = queries[i][0], y = queries[i][1];
            b[i] = dp[y] >= y-x+1;
        }
        return b;
    }

    /**
     * 322. 零钱兑换
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，
     * 表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有
     * 任何一种硬币组合能组成总金额，返回 -1 。
     * 你可以认为每种硬币的数量是无限的。
     * @param coins
     * @param amount
     * @return
     */
    public static int coinChange(int[] coins, int amount) {
       int max = amount + 1;
       int[] dp = new int[max];
       Arrays.fill(dp, max);
       dp[0] = 0;
        for (int i = 0; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i-coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    /**
     * 111. 二叉树的最小深度
     * 给定一个二叉树，找出其最小深度。
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * 说明：叶子节点是指没有子节点的节点。
     * @param root
     * @return
     */
    public static int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if(root.left != null && root.right != null) {
            return Math.min(left, right) + 1;
        }
        return Math.max(left, right) + 1;
    }

    /**
     * 169. 多数元素
     * 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于
     * ⌊ n/2 ⌋ 的元素.你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     * @param nums
     * @return
     */
    public static int majorityElement(int[] nums) {
//        int n = nums.length;
//        for (int i = 0; i < n; i++) {
//            int k = 1;
//            for (int j = i + 1; j < n; j++) {
//                if(nums[i] == nums[j]) {
//                    k++;
//                }
//            }
//            if (k>n/2) {
//                return nums[i];
//            }
//        }
//        return 0;
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    /**
     * 3117. 划分数组得到最小的值之和
     * 给你两个数组 nums 和 andValues，长度分别为 n 和 m。
     * 数组的值等于该数组的 最后一个 元素。你需要将 nums 划分为 m 个 不相交的连续
     * 子数组，对于第 ith 个子数组 [li, ri]，子数组元素的按位 AND 运算结果等于
     * andValues[i]，换句话说，对所有的 1 <= i <= m，
     * nums[li] & nums[li + 1] & ... & nums[ri] == andValues[i] ，
     * 其中 & 表示按位 AND 运算符。
     * 返回将 nums 划分为 m 个子数组所能得到的可能的 最小 子数组 值 之和。
     * 如果无法完成这样的划分，则返回 -1 。
     * @param nums
     * @param andValues
     * @return
     */
    public int minimumValueSum(int[] nums, int[] andValues) {
        Map<Long, Integer> memo = new HashMap<>();
        int ans = dfs(0, 0, -1, nums, andValues, memo);
        return ans < Integer.MAX_VALUE / 2 ? ans : -1;
    }

    private int dfs(int i, int j, int and, int[] nums, int[] andValues, Map<Long, Integer> memo) {
        int n = nums.length;
        int m = andValues.length;
        if (n - i < m - j) {
            return Integer.MAX_VALUE / 2;
        }
        if (j == m){
            return i == n ? 0 : Integer.MAX_VALUE / 2;
        }
        and &= nums[i];
        long mask = (long) i << 36 | (long) j << 32 | and;
        if (memo.containsKey(mask)) {
            return memo.get(mask);
        }
        int res = dfs(i + 1, j, and, nums, andValues, memo);
        if (and == andValues[j]) {
            res = Math.min(res, dfs(i+1, j+1, -1, nums, andValues, memo) + nums[i]);
        }
        memo.put(mask, res);
        return res;
    }

    public static int minimumOperationsToMakeKPeriodic(String word, int k) {
        return 0;
    }

    /**
     * 552. 学生出勤记录 II
     * 可以用字符串表示一个学生的出勤记录，其中的每个字符用来标记当天
     * 的出勤情况（缺勤、迟到、到场）。记录中只含下面三种字符：
     *     'A'：Absent，缺勤
     *     'L'：Late，迟到
     *     'P'：Present，到场
     * 如果学生能够 同时 满足下面两个条件，则可以获得出勤奖励：
     *     按总出勤计，学生缺勤（'A'）严格少于两天。
     *     学生不会存在连续3天或连续3天以上的迟到（'L'）记录。
     * 给你一个整数n,表示出勤记录的长度（次数）。
     * 请你返回记录长度为n时，可能获得出勤奖励的记录情况数量 。
     * 答案可能很大，所以返回对 109 + 7 取余 的结果。
     * @param n
     * @return
     */
//    private static final int MOD = 1_000_000_007;
    private static final int MX = 100_001;
    private static final int[][][] memo = new int[MX][2][3];
    public static int checkRecord(int n) {
        return dfs(n, 0, 0);
    }
    // i:代表有多少数,j:代表有多少个A,k:代表有多少个连续的L。
    private static int dfs(int i, int j, int k) {
        if (i == 0) {
            return 1;
        }
        if (memo[i][j][k] > 0) {
            return memo[i][j][k];
        }
        // 如果填入P,则i-1, j个A, L不连续了，所以清零
        long res = dfs(i-1, j, 0);
        if (j < 1) {
            // 如果填入A, 则i-1, 有1个A了, L不连续了，所以清零
            res += dfs(i-1, 1, 0);
        }
        if (k < 2) {
            // 如果填入L, 则i-1, j个A, k加一
            res += dfs(i-1, j, k+1);
        }
        return memo[i][j][k] = (int) res%MOD;
    }

    /**
     * 377. 组合总和 Ⅳ
     * 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。
     * 请你从 nums 中找出并返回总和为 target 的元素组合的个数。
     * 题目数据保证答案符合 32 位整数范围。
     * @param nums
     * @param target
     * @return
     */
    public static int combinationSum4(int[] nums, int target) {
        int[] memo = new int[target + 1];
        Arrays.fill(memo, -1);
        return dfs1(memo, nums, target);
    }
    public static int dfs1(int[] memo, int[] nums, int target) {
        if (target == 0) {
            return 1;
        }
        if (memo[target] != -1) {
            return memo[target];
        }
        int res = 0;
        for (int num : nums) {
            if (num <= target) {
                res += dfs1(memo,nums, target-num);
            }
        }
        return memo[target] = res;
    }

    /**
     * 2466. 统计构造好字符串的方案数
     * 给你整数 zero ，one ，low 和 high ，我们从空字符串开始构造一个字符串，
     * 每一步执行下面操作中的一种：
     *     将 '0' 在字符串末尾添加 zero 次。
     *     将 '1' 在字符串末尾添加 one 次。
     * 以上操作可以执行任意次。
     * 如果通过以上过程得到一个长度在low和high之间（包含上下边界）的字符串，
     * 那么这个字符串我们称为好字符串。
     * 请你返回满足以上要求的不同好字符串数目。由于答案可能很大，
     * 请将结果对109 + 7 取余 后返回。
     * @param low
     * @param high
     * @param zero
     * @param one
     * @return
     */
//    private static final int MOD = 1_000_000_007;
    public static int countGoodStrings(int low, int high, int zero, int one) {
        int[] memo = new int[high+1];
        Arrays.fill(memo, -1);
        return dfs2(memo, 0, low, high, zero, one);
    }
    public static int dfs2(int[] memo, int i, int low, int high, int zero, int one) {
        if (i > high) {
            return 0;
        }
        if (memo[i] != -1) {
            return memo[i];
        }
        int res = 0;
        if (i <= high && i >= low) {
            res ++;
        }
        res = (res+dfs2(memo, i+zero, low, high, zero, one)) % MOD;
        res = (res+dfs2(memo, i+one, low, high, zero, one)) % MOD;
        return memo[i] = res;
    }

    /**
     * 3154. 到达第 K 级台阶的方案数
     * 给你有一个 非负 整数 k 。有一个无限长度的台阶，最低 一层编号为 0 。
     * Alice有一个整数jump,一开始值为0。Alice从台阶1开始，可以使用任意次操作，
     * 目标是到达第k级台阶。假设Alice位于台阶i,一次操作中,Alice可以：
     * 向下走一级到i-1,但该操作不能连续使用，如果在台阶第0级也不能使用。
     * 向上走到台阶 i + 2^jump 处，然后jump变为jump + 1 。
     * 请你返回 Alice 到达台阶 k 处的总方案数。
     * 注意,Alice可能到达台阶k处后，通过一些操作重新回到台阶k处，这视为不同的方案。
     * @param k
     * @return
     */
    public static int waysToReachStair(int k) {
        return dfs3(1, 0, 0, k, new HashMap<>());
    }

    public static int dfs3(int i, int j, int preDown, int k, Map<Long, Integer> memo) {
        if (i > k + 1) {
            return 0;
        }
        // 把状态 (i, j, preDown) 压缩成一个long,作为标识符
        long mask = (long) i << 32 | j << 1 | preDown;
        if (memo.containsKey(mask)) {
            return memo.get(mask);
        }
        int res = i == k ? 1 : 0;
        res += dfs3(i + (1<<j), j+1, 0, k, memo);
        if (preDown == 0 && i > 0) {
            res += dfs3(i-1, j, 1, k, memo);
        }
        memo.put(mask, res);
        return res;
    }

    /**
     * 2266. 统计打字方案数
     * Alice 在给 Bob 用手机打字。数字到字母的 对应 如下图所示。
     * 为了打出一个字母,Alice需要按对应字母i次，i是该字母在这个按键上所处的位置。
     * 比方说,为了按出字母's',Alice需要按'7'四次.类似的，Alice需要按'5'两次得到字母'k' 。
     * 注意，数字'0'和'1'不映射到任何字母，所以Alice不使用它们。
     * 但是，由于传输的错误，Bob没有收到Alice打字的字母信息,反而收到了按键的字符串信息。
     * 比方说，Alice 发出的信息为 "bob" ，Bob 将收到字符串 "2266622" 。
     * 给你一个字符串pressedKeys,表示Bob收到的字符串,请你返回Alice总共可能发出多少种文字信息。
     * 由于答案可能很大，将它对 109 + 7 取余 后返回。
     * @param pressedKeys
     * @return
     */
    private static final int MOD = 1_000_000_007;
    private static final int[] f = new int[MX], g = new int[MX];
    static {
        f[0] = g[0] = 1;
        f[1] = g[1] = 1;
        f[2] = g[2] = 2;
        f[3] = g[3] = 4;
        for (int i = 4; i < MX; i++) {
            f[i] = (int) (((long) f[i-1] + f[i-2] + f[i-3]) % MOD);
            // 7 / 9
            g[i] = (int) (((long) g[i-1] + g[i-2] + g[i-3] + g[i-4]) % MOD);
        }
    }
    public int countTexts(String s) {
        int ans = 1, cnt = 0;
        for (int i = 0; i < s.length(); i++) {
            cnt++;
            int c = s.charAt(i);
            if (i == s.length() - 1 || c != s.charAt(i + 1)) {
                ans = (int) ((long) ans * (c != '7' && c != '9' ? f[cnt] : g[cnt]) % MOD);
                cnt = 0;
            }
        }
        return ans;
    }

    /**
     * 3007. 价值和小于等于 K 的最大数字
     * 给你一个整数 k 和一个整数 x 。整数 num 的价值是它的二进制表示中在 x，2x，3x等位置处
     * 设置位的数目（从最低有效位开始）。下面的表格包含了如何计算价值的例子。
     * x	num	Binary Representation	Price
     * 1	13	000001101	3
     * 2	13	000001101	1
     * 2	233	011101001	3
     * 3	13	000001101	1
     * 3	362	101101010	2
     * num的累加价值是从1到num的数字的总价值.如果num的累加价值小于或等于k则被认为是廉价的。
     * 请你返回 最大 的廉价数字
     * @param k
     * @param x
     * @return
     */
    public long findMaximumNumber(long k, int x) {
        return 0;
    }

    /**
     * 3133. 数组最后一个元素的最小值
     * 给你两个整数n和x.你需要构造一个长度为n的正整数数组nums,对于所有0<=i<n-1,
     * 满足nums[i+1]大于nums[i],并且数组nums中所有元素的按位AND运算结果为x。
     * 返回nums[n-1]可能的最小值。
     * @param n
     * @param x
     * @return
     */
    public static long minEnd(int n, int x) {
//        if (n==1) {
//            return x;
//        }
//        long[] a = new long[n+1];
//        a[0] = x;
//        int m = 1;
//        for (long i = x+1; i < 1000000000000000000L; i++) {
//            long k = a[0];
//            k = k & i;
//            if (k == x) {
//                a[m++] = i;
//                if (m == n) {
//                    break;
//                }
//            }
//        }
//        return a[n-1];

        n--;
        long ans = x;
        int i = 0, j = 0;
        while ((n >> j) > 0) {
            if ((ans >> i & 1) == 0) {
                ans |= (long) (n >> j & 1) << i;
                j++;
            }
            i++;
        }
        return ans;
    }


// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};

    /**
     * 690. 员工的重要性
     * 你有一个保存员工信息的数据结构，它包含了员工唯一的 id ，重要度和直系下属的 id 。
     * 给定一个员工数组 employees，其中：
     *     employees[i].id 是第 i 个员工的 ID。
     *     employees[i].importance 是第 i 个员工的重要度。
     *     employees[i].subordinates 是第 i 名员工的直接下属的 ID 列表。
     * 给定一个整数 id表示一个员工的 ID，返回这个员工和他所有下属的重要度的总和。
     * @param employees
     * @param id
     * @return
     */
    public int getImportance(List<Employee> employees, int id) {
        Map<Integer, Employee> employeeMap = new HashMap<>(employees.size());
        for (Employee employee : employees) {
            employeeMap.put(employee.id, employee);
        }
        return dfs4(employeeMap, id);
    }
    public int dfs4(Map<Integer, Employee> employeeMap, int id) {
        Employee e = employeeMap.get(id);
        int res = e.importance;
        for (int subordinate : e.subordinates) {
            res += dfs4(employeeMap, subordinate);
        }
        return res;
    }

    /**
     * 547. 省份数量
     * 有n个城市，其中一些彼此相连，另一些没有相连。如果城市a与城市b直接相连，且
     * 城市b与城市c直接相连，那么城市a与城市c间接相连。
     * 省份是一组直接或间接相连的城市，组内不含其他没有相连的城市。
     * 给你一个nxn的矩阵isConnected，其中 isConnected[i][j]=1表示
     * 第i个城市和第j个城市直接相连，而isConnected[i][j]=0表示二者不直接相连。
     * 返回矩阵中省份的数量。
     * @param isConnected
     * @return
     */
    public int findCircleNum(int[][] isConnected) {
        return 0;
    }

    /**
     * 1971. 寻找图中是否存在路径
     * 有一个具有n个顶点的双向图，其中每个顶点标记从0到n-1(包含0和n-1).
     * 图中的边用一个二维整数数组edges表示，其中edges[i]=[ui,vi]表示
     * 顶点ui和顶点vi之间的双向边。每个顶点对由最多一条边连接，
     * 并且没有顶点存在与自身相连的边。
     * 请你确定是否存在从顶点source开始，到顶点destination结束的有效路径 。
     * 给你数组edges和整数n、source和destination，如果从source到
     * destination存在有效路径，则返回true，否则返回false。
     * @param n
     * @param edges
     * @param source
     * @param destination
     * @return
     */
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        if (source == destination) {
            return true;
        }
        UnionFind uf = new UnionFind(n);
        for (int[] edge : edges) {
            uf.uni(edge[0], edge[1]);
        }
        return uf.connect(source, destination);
    }

    class UnionFind {
        private int[] parent;
        private int[] rank;

        public UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        public void uni(int x, int y) {
            int rootx = find(x);
            int rooty = find(y);
            if (rootx != rooty) {
                if (rank[rootx] > rank[rooty]) {
                    parent[rooty] = rootx;
                } else if (rank[rootx] < rank[rooty]) {
                    parent[rootx] = rooty;
                } else {
                    parent[rooty] = rootx;
                    rank[rootx]++;
                }
            }
        }

        public int find(int a) {
            if (parent[a] != a) {
                parent[a] = find(parent[a]);
            }
            return parent[a];
        }

        public boolean connect(int x, int y) {
            return find(x) == find(y);
        }
    }

    /**
     * 3134. 找出唯一性数组的中位数
     * 给你一个整数数组nums.数组nums的唯一性数组是一个按元素从小到大
     * 排序的数组，包含了nums的所有非空子数组中不同元素的个数。
     * 换句话说，这是由所有0<=i<=j<nums.length的
     * distinct(nums[i..j])组成的递增数组。
     * 其中,distinct(nums[i..j])表示从下标i到下标j的子数组中
     * 不同元素的数量。返回nums唯一性数组的中位数 。
     * 注意，数组的中位数定义为有序数组的中间元素。如果有两个中间元素，则取值较小的那个。
     * @param nums
     * @return
     */
    public int medianOfUniquenessArray(int[] nums) {
        int n = nums.length;
        long k = ((long) n * (n + 1) / 2 + 1) / 2; //中位数
        int left = 0;
        int right = n;
        while (left + 1 < right) {
            int mid = (left + right) / 2;
            if (find1(nums, mid, k)) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return right;
    }

    public static boolean find1(int[] nums, int upper, long k){
        HashMap<Integer, Integer> map= new HashMap<>();
        int l = 0;
        long cnt = 0;
        for (int r = 0; r < nums.length; r++) {
            map.merge(nums[r], 1, Integer::sum);
            while (map.size() > upper) {
                int out = nums[l++];
                if (map.merge(out, -1, Integer::sum) == 0) {
                    map.remove(out);
                }
            }
            cnt += r - l + 1;
            if (cnt >= k) {
                return true;
            }
        }
        return false;
    }

    /**
     * 740. 删除并获得点数
     * 给你一个整数数组 nums ，你可以对它进行一些操作。
     * 每次操作中,选择任意一个nums[i],删除它并获得nums[i]的点数。
     * 之后，你必须删除所有等于nums[i]-1和nums[i]+1的元素。
     * 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
     * @param nums
     * @return
     * nums = [2,2,3,3,3,4]
     * arr = [0,0,4,9,4]
     * index  0 1 2 3 4  = > 对新数组进行打家劫舍
     */
    public static int deleteAndEarn(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int[] a = new int[10001];
        int i;
        for (i = 0; i < n; i++) {
            if (a[nums[i]] == 0) {
                a[nums[i]] = nums[i];
            }
            for (int j = i+1; j < n; j++) {
                if (nums[i] == nums[j]) {
                    a[nums[i]] += nums[i];
                    i++;
                }
            }
        }
        return dfs6(a);
    }

    public static int dfs6(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
        }
        return dp[n-1];
    }

    /**
     * 26. 删除有序数组中的重复项  双指针
     * 给你一个非严格递增排列的数组nums,请你原地删除重复出现的元素,使每个元素只出现一次,
     * 返回删除后数组的新长度.元素的相对顺序应该保持一致.然后返回nums中唯一元素的个数。
     * 考虑nums的唯一元素的数量为k,你需要做以下事情确保你的题解可以被通过：
     * 更改数组nums,使nums的前k个元素包含唯一元素,并按照它们最初在nums中出现的顺序排列.
     * nums的其余元素与nums的大小不重要。返回k。
     * @param nums
     * @return
     */
    public static int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int fast = 1, slow = 1;
        while (fast < n) {
            if (nums[fast-1] != nums[fast]) {
                nums[slow] = nums[fast];
                 ++slow;
            }
            ++fast;
        }
        System.out.println(Arrays.toString(nums));
        return slow;
    }

    /**
     * 3144. 分割字符频率相等的最少子字符串
     * 给你一个字符串s,你需要将它分割成一个或者更多的平衡子字符串.比方说,s=="ababcc"
     * 那么("abab","c","c"),("ab","abc","c")和("ababcc")都是合法分割，
     * 但是("a","bab","cc"),("aba","bc","c")和 ("ab","abcc")不是,
     * 不平衡的子字符串用粗体表示。请你返回s最少能分割成多少个平衡子字符串。
     * 注意：一个平衡字符串指的是字符串中所有字符出现的次数都相同。
     * @param s
     * @return
     */
    public static int minimumSubstringsInPartition(String s) {
        char[] s1 = s.toCharArray();
        int n = s1.length;
        int[] memo = new int[n];
        return dfs7(memo, n-1, s1);
    }

    public static int dfs7(int[] memo, int i, char[] s) {
        if (i<0) {
            return 0;
        }
        if(memo[i] > 0) {
            return memo[i];
        }
        int res = Integer.MAX_VALUE;
        int[] cnt = new int[26]; //存储26个英文字母出现的次数
        int k = 0, maxCnt = 0; //maxCnt 记录出现次数最多的字符的频次
        for (int j = i; j >= 0; j--) {
            //更新当前子字符串中不同字符的数量 k
            k += cnt[s[j] - 'a']++ == 0 ? 1 : 0;
            maxCnt = Math.max(maxCnt, cnt[s[j] - 'a']);
            if (i - j + 1 == k * maxCnt) {
                res = Math.min(res, dfs7(memo,j-1, s) + 1);
            }
        }
        memo[i] = res;
        return res;
    }

    /**
     * 3142. 判断矩阵是否满足条件
     * 给你一个大小为mxn的二维矩阵grid.你需要判断每一个格子grid[i][j]是否满足：
     * 如果它下面的格子存在,那么它需要等于它下面的格子,也就是grid[i][j]==grid[i+1][j]。
     * 如果它右边的格子存在,那么它需要不等于它右边的格子,也就是grid[i][j]!=grid[i][j+1]。
     * 如果所有格子都满足以上条件，那么返回true，否则返回false。
     * @param grid
     * @return
     */
    public static boolean satisfiesConditions(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean flag = true;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i+1<m && grid[i][j] != grid[i + 1][j]) {
                    flag = false;
                }
                if (j+1<n &&  grid[i][j] == grid[i][j + 1] ) {
                    flag = false;
                }
            }
        }
        return flag;
    }

    /**
     * 415. 字符串相加
     * 给定两个字符串形式的非负整数num1和num2,计算它们的和并同样以字符串形式返回。
     * 你不能使用任何內建的用于处理大整数的库(比如 BigInteger),也不能直接将输入的字符串转换为整数形式。
     * @param num1
     * @param num2
     * @return
     */
//    public static String addStrings(String num1, String num2) {
//        int n1 = num1.length();
//        int n2 = num2.length();
//        int n3 = Math.max(n1, n2);
//        long n = 0;
//        for (int i = 0; i < n3; i++) {
//            if (i+1>n1) {
//                n += (long)(num2.charAt(n2-1-i) - '0') * Math.pow(10,i);
//            } else if (i+1>n2) {
//                n += (long)(num1.charAt(n1-1-i) - '0') * Math.pow(10,i);
//                System.out.println((num1.charAt(n1-1-i) - '0') * Math.pow(10,i));
//            } else if (i>0){
//                n += (long)(num1.charAt(n1-1-i) - '0'+ num2.charAt(n2-1-i) - '0') * Math.pow(10,i);
//            } else {
//                n += num2.charAt(n2-1-i) - '0' + num1.charAt(n1-1-i) - '0';
//            }
//        }
//        return String.valueOf(n);
//    }
    public static String addStrings(String num1, String num2) {
        int n1 = num1.length();
        int n2 = num2.length();
        int maxLen = Math.max(n1, n2);
        StringBuilder result = new StringBuilder();
        int carry = 0;

        for (int i = 0; i < maxLen; i++) {
            int digit1 = (i < n1) ? num1.charAt(n1 - 1 - i) - '0' : 0;
            int digit2 = (i < n2) ? num2.charAt(n2 - 1 - i) - '0' : 0;

            int sum = digit1 + digit2 + carry;
            carry = sum / 10;
            result.append(sum % 10);
        }

        if (carry > 0) {
            result.append(carry);
        }

        return result.reverse().toString();
    }

    /**
     * 3153. 所有数对中数位不同之和
     * 你有一个数组nums，它只包含正整数，所有正整数的数位长度都相同 。
     * 两个整数的数位不同指的是两个整数相同位置上不同数字的数目。
     * 请你返回nums中所有整数对里，数位不同之和。
     * @param nums
     * @return
     */
    public static long sumDigitDifferences(int[] nums) {
        //超时
//        int n = nums.length;
//        int m = String.valueOf(nums[0]).length();
//        int ans = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = i+1; j < n; j++) {
//                String si = String.valueOf(nums[i]);
//                String sj = String.valueOf(nums[j]);
//                for (int k = 0; k < m; k++) {
//                    if (si.charAt(k) != sj.charAt(k)) {
//                        ans++;
//                    }
//                }
//            }
//        }
//        return ans;
        // 拆位法+枚举右维护左
        long ans = 0;
        int[][] cnt = new int[Integer.toString(nums[0]).length()][10];
        for (int k = 0; k < nums.length; k++) {
            int x = nums[k];
            for (int i = 0; x > 0; x/=10, i++) {
                ans += k - cnt[i][x%10] ++;
            }
        }
        return ans;
    }

    /**
     * 912. 排序数组
     * 给你一个整数数组 nums，请你将该数组升序排列。
     * @param nums
     * @return
     */
    private static final Random RANDOM = new Random();
    public static int[] sortArray(int[] nums) {
        QuickSort(nums, 0 ,nums.length-1);
//        int n = nums.length;
//        for (int i = 0; i < n; i++) {
//            for (int j = i+1; j < n; j++) {
//                if (nums[j] < nums[i]) {
//                    int temp = nums[i];
//                    nums[i]  = nums[j];
//                    nums[j]  = temp;
//                }
//            }
//        }
        return nums;
    }

    //根据base将数组分为两块，小于base都放到左边，大于base都放到右边。然后递归排序
    // 根据基准将数组分为两块，小于基准的都放到左边，大于基准的都放到右边，然后递归排序
    public static void QuickSort(int[] a, int left, int right) {
        if (left >= right) {
            return;
        }

        // 选择一个随机基准
        int pivotIndex = RANDOM.nextInt(right - left + 1) + left;
        int pivot = a[pivotIndex];

        // 交换基准元素到最左边
        swap(a, left, pivotIndex);

        int l = left;
        int r = right;

        while (l < r) {
            while (l < r && a[r] >= pivot) {
                r--;
            }
            while (l < r && a[l] <= pivot) {
                l++;
            }
            if (l < r) {
                swap(a, l, r);
            }
        }

        // 把基准元素放到正确的位置
        swap(a, left, l);

        // 递归排序左右部分
        QuickSort(a, left, l - 1);
        QuickSort(a, l + 1, right);
    }

    // 交换数组中两个元素
    private static void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    /**
     * 1234. 替换子串得到平衡字符串
     * 有一个只含有 'Q', 'W', 'E', 'R' 四种字符，且长度为 n 的字符串。
     * 假如在该字符串中，这四个字符都恰好出现 n/4 次，那么它就是一个「平衡字符串」。
     * 给你一个这样的字符串 s，请通过「替换一个子串」的方式，使原字符串 s 变成一个「平衡字符串」。
     * 你可以用和「待替换子串」长度相同的 任何 其他字符串来完成替换。
     * 请返回待替换子串的最小可能长度。
     * 如果原字符串自身就是一个平衡字符串，则返回 0。
     * @param s
     * @return
     */
    public static int balancedString(String s) {
        return 0;
    }

    // 生成所有子数组
    public static List<int[]> getAllSubarrays(int[] arr) {
        List<int[]> subarrays = new ArrayList<>();
        for (int start = 0; start < arr.length; start++) {
            for (int end = start + 1; end <= arr.length; end++) {
                int[] subarray = new int[end - start];
                System.arraycopy(arr, start, subarray, 0, end - start);
                subarrays.add(subarray);
            }
        }
        return subarrays;
    }

    // 计算子数组的最大乘积
    public static int maxSubarrayProduct(int[] arr) {
        List<int[]> allSubarrays = getAllSubarrays(arr);
        int maxProduct = Integer.MIN_VALUE;

        for (int[] subarray : allSubarrays) {
            int product = 1;
            for (int num : subarray) {
                product *= num;
            }
            maxProduct = Math.max(maxProduct, product);
        }

        return maxProduct;
    }

    public static void main(String[] args) {
        int n = 5;

    }
}
