package com.wr.Mouth7;

import java.util.*;

/**
 * ClassName: Solution
 * Description:
 * date: 2024/7/2 12:10
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Solution {
    /**
     * 3115. 质数的最大距离
     * 给你一个整数数组 nums:
     * 返回两个（不一定不同的）质数在 nums 中 下标 的 最大距离。
     * @param nums
     * @return
     */
    public static int maximumPrimeDifference(int[] nums) {
        int[] a = new int[nums.length];
        int k =0;
        for(int i =0; i<nums.length;i++) {
            int j =2;
            for(; j<nums[i]; j++) {
                if(nums[i]%j==0) {
                    break;
                }
            }
            if(nums[i]==j){
                a[k++] = i;
            }
        }
        int r = maxMin(a);
        return r;
    }

    public static int maxMin(int[] a) {
        int max=a[0];
        int min=a[0];
        for(int i = 0;i<a.length;i++) {
            if(a[i]>max) {
                max = a[i];
            }
            if(a[i]<min&&a[i]!=0) {
                min = a[i];
            }
        }
        return max-min;
    }

    /**
     * 3099. 哈沙德数
     * 如果一个整数能够被其各个数位上的数字之和整除，则称之为 哈沙德数（Harshad number）。
     * 给你一个整数 x 。如果 x 是 哈沙德数 ，则返回 x 各个数位上的数字之和，否则，返回 -1 。
     * @param x
     * @return
     */
    public static int sumOfTheDigitsOfHarshadNumber(int x) {
        int a = x;
        int k = 0;
        for(int i=0; i<3;i++) {
            k+=x%10;
            x/=10;
        }
        if(a%k == 0) {
            return k;
        }
        return -1;
    }

    /**
     * 3086. 拾起 K 个 1 需要的最少行动次数
     * 给你一个下标从 0 开始的二进制数组 nums，其长度为 n ；另给你一个 正整数 k 以及一个 非负整数 maxChanges 。
     * Alice 在玩一个游戏，游戏的目标是让 Alice 使用 最少 数量的 行动 次数从 nums 中拾起 k 个 1 。游戏开始时，
     * Alice 可以选择数组 [0, n - 1] 范围内的任何索引 aliceIndex 站立。如果 nums[aliceIndex] == 1 ，Alice 会拾起一个 1 ，
     * 并且 nums[aliceIndex] 变成0（这不算作一次行动）。之后，Alice 可以执行任意数量 的 行动（包括零次），
     * 在每次行动中 Alice 必须 恰好 执行以下动作之一：
     * 选择任意一个下标 j != aliceIndex 且满足 nums[j] == 0 ，然后将 nums[j] 设置为 1 。这个动作最多可以执行 maxChanges 次。
     * 选择任意两个相邻的下标 x 和 y（|x - y| == 1）且满足 nums[x] == 1, nums[y] == 0 ，
     * 然后交换它们的值（将 nums[y] = 1 和 nums[x] = 0）。如果 y == aliceIndex，在这次行动后 Alice 拾起一个 1 ，并且 nums[y] 变成 0 。
     * 返回 Alice 拾起 恰好 k 个 1 所需的 最少 行动次数。
     * @param nums
     * @param k
     * @param maxChanges
     * @return
     */
    public static long minimumMoves(int[] nums, int k, int maxChanges) {
        List<Integer> pos = new ArrayList<>();
        int c = 0;
        //maxChanges较大
        //对于连续的1可以直接通过一次操作得到，其他的需要通过两次操作得到。
        for (int i = 0; i<nums.length; i++) {
            if(nums[i] == 0) {
                continue;
            }
            pos.add(i);
            c = Math.max(c,1);
            if(i>0 && nums[i-1] == 1) {
                if (i>1 && nums[i-2] == 1) {
                    c = 3; //有3个1
                } else {
                    c = Math.max(c,2);//有2个连续的1
                }
            }
        }
        c = Math.min(c,k);
        if(maxChanges >= k-c) {
            //其余k-c个1可以全部用两次操作得到
            return Math.max(c-1, 0) + (k-c) * 2;
        }
        //maxChanges = 0
        int n = pos.size();
        long[] sum = new long[n+1];
        for(int i = 0; i<n; i++) {
            sum[i+1] = sum[i] + pos.get(i);
        }
        //maxChanges较小
        long ans = Long.MAX_VALUE;
        int size = k-maxChanges;
        for (int right = size; right<=n; right++) {
            int left = right-size;
            int i = left+size/2;
            long index = pos.get(i);
            long s1 = index * (i-left) - (sum[i] - sum[left]);
            long s2 = sum[right] - sum[i] - index * (right-1);
            ans = Math.min(ans,s1+s2);
        }
        return ans + maxChanges * 2;
    }

    /**
     * 3033. 修改矩阵
     * 给你一个下标从0开始、大小为 m x n的整数矩阵matrix，
     * 新建一个下标从0开始、名为answer的矩阵。使answer与matrix相等,
     * 接着将其中每个值为-1的元素替换为所在列的最大元素
     * @param matrix
     * @return
     */
    public static int[][] modifiedMatrix(int[][] matrix) {
        for (int i = 0; i<matrix[0].length; i++) {
            int max = matrix[0][i];
            for (int j = 0; j<matrix.length; j++) {
                if (matrix[j][i] > max) {
                    max = matrix[j][i];
                }
            }
            for (int j = 0; j<matrix.length; j++) {
                if (matrix[j][i] == -1) {
                    matrix[j][i] = max;
                }
            }
        }
        return matrix;
    }

    /**
     * 724. 寻找数组的中心下标
     * 给你一个整数数组 nums ，请计算数组的 中心下标 。
     * 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。
     * 如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。
     * 这一点对于中心下标位于数组最右端同样适用。
     * 如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。
     * @param nums
     * @return
     */
    public static int pivotIndex(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int left  = 0, right = 0;
            for (int j = 0; j < nums.length; j++) {
                if (j<i) {
                    left += nums[j];
                }
                if (j>i) {
                    right += nums[j];
                }
            }
            if (left == right) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 3102. 最小化曼哈顿距离
     * 给你一个下标从0开始的数组 points ，它表示二维平面上一些点的整数坐标，
     * 其中 points[i] = [xi, yi] 。两点之间的距离定义为它们的曼哈顿距离。
     * 请你恰好移除一个点，返回移除后任意两点之间的最大距离可能的最小值。
     * @param points
     * @return
     */
    public static int minimumDistance(int[][] points) {
        TreeMap<Integer, Integer> xs = new TreeMap<>();
        TreeMap<Integer, Integer> ys = new TreeMap<>();
        // |x1-x2|+|y1-y2|=max(|x1'-x2'|,|y1'-y2'|)
        for (int[] p: points) {
            xs.merge(p[0] + p[1], 1, Integer::sum);
            ys.merge(p[1] - p[0], 1, Integer::sum);
        }
        int ans = Integer.MAX_VALUE;
        for (int[] p : points) {
            int x = p[0] + p[1];
            int y = p[1] - p[0];
            if (xs.get(x) == 1) {
                xs.remove(x);
            } else {
                xs.merge(x, -1, Integer::sum);
            }
            if (ys.get(y) == 1) {
                ys.remove(y);
            } else {
                ys.merge(y, -1, Integer::sum);
            }
            int dx = xs.lastKey() - xs.firstKey();
            int dy = ys.lastKey() - ys.firstKey();
            ans = Math.min(ans, Math.max(dx, dy));

            xs.merge(x, 1, Integer::sum);
            ys.merge(y, 1, Integer::sum);

        }

        return ans;
    }

    /**
     * 2970. 统计移除递增子数组的数目
     * 给你一个下标从 0 开始的 正 整数数组 nums 。
     * 如果 nums 的一个子数组满足：移除这个子数组后剩余元素 严格递增 ，
     * 那么我们称这个子数组为 移除递增 子数组。比方说，[5, 3, 4, 6, 7]中的
     * [3, 4] 是一个移除递增子数组，因为移除该子数组后，[5, 3, 4, 6, 7]
     * 变为 [5, 6, 7] ，是严格递增的。请你返回 nums 中 移除递增 子数组的总数目。
     * 注意 ，剩余元素为空的数组也视为是递增的。子数组 指的是一个数组中一段连续的元素序列。
     * @param nums
     * @return
     */
    public static int incremovableSubarrayCount(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 1; j < nums.length; j++) {
                if (isIncreasing(nums, i, j)) {
                    res ++;
                }
            }
        }
        return res;
    }

    public static boolean isIncreasing(int[] nums, int l, int r) {
        for (int i = 1; i < nums.length; i++) {
            if (i >= 1 && i <= r + 1) {
                continue;
            }
            if (nums[i] <= nums[i-1]) {
                return false;
            }
        }
        if (l-1 >= 0 && r + 1 < nums.length && nums[r+1] <= nums[l-1]) {
            return false;
        }
        return true;
    }

    //双指针
    public static long incremovableSubarrayCount1(int[] nums) {
        long ans = 0;
        int len = nums.length;
        int l = 0;
        while (l < len - 1) {
            if (nums[l] >= nums[l + 1]) {
                break;
            }
            l++;
        }
        if (l == len - 1) {
            return 1L * len * (len + 1) / 2;
        }

        ans += l + 2;
        for (int r = len - 1; r > 0; r--) {
            if (r < len - 1 && nums[r] >= nums[r + 1]) {
                break;
            }

            while (l >= 0 && nums[l] >= nums[r]) {
                l--;
            }
            ans += l + 2;
        }

        return ans;
    }

    public static int[] numberGame(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i+=2) {
            int a = nums[i];
            nums[i] = nums[i+1];
            nums[i+1] = a;
        }
        return nums;
    }

    /**
     * 3112. 访问消失节点的最少时间
     * 给你一个二维数组 edges 表示一个 n 个点的无向图，其中 edges[i] = [ui, vi, lengthi]
     * 表示节点 ui 和节点 vi 之间有一条需要 lengthi 单位时间通过的无向边。
     * 同时给你一个数组 disappear ，其中 disappear[i] 表示节点 i 从图中消失的时间点，
     * 在那一刻及以后，你无法再访问这个节点。注意，图有可能一开始是不连通的，两个节点之间也可能有多条边。
     * 请你返回数组 answer ，answer[i] 表示从节点 0 到节点 i 需要的 最少 单位时间。
     * 如果从节点 0 出发 无法 到达节点 i ，那么 answer[i] 为 -1 。
     * @param n
     * @param edges
     * @param disappear
     * @return
     */
    public static int[] minimumTime(int n, int[][] edges, int[] disappear) {
        int[] a = new int[disappear.length];
        for (int i = 0; i < n; i++) {
            if (edges[i][0] == 0) {
                if (edges[i][2] < disappear[edges[i][1]]) {
                    a[edges[i][1]] = edges[i][2];
                } else {
                    a[edges[i][1]] = -1;
                }
            }
        }
        return a;
    }

    public static int maximumDetonation(int[][] bombs) {
        int n = bombs.length;
        Map<Integer, List<Integer>> edges = new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && isConnected(bombs, i, j)) {
                    edges.putIfAbsent(i, new ArrayList<Integer>());
                    edges.get(i).add(j);
                }
            }
         }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            boolean[] visited = new boolean[n];
            int cnt = 1;
            Queue<Integer> queue = new ArrayDeque<Integer>();
            queue.offer(i);
            visited[i] = true;
            while (!queue.isEmpty()) {
                int cidx = queue.poll();
                for (int nidx : edges.getOrDefault(cidx, new ArrayList<Integer>())) {
                    if (visited[nidx]) {
                        continue;
                    }
                    ++cnt;
                    queue.offer(nidx);
                    visited[nidx] = true;
                }
            }
            res = Math.max(res, cnt);
        }
        return res;
    }

    public static boolean isConnected(int[][] bombs, int u, int v) {
        long dx = bombs[u][0] - bombs[v][0];
        long dy = bombs[u][1] - bombs[v][1];
        return (long) bombs[u][2] * bombs[u][2] >= dx * dx + dy * dy;
    }

    public static List<Integer> relocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int i = 0; i< moveFrom.length; i++) {
            set.remove(moveFrom[i]);
            set.add(moveTo[i]);
        }
        List<Integer> ans = new ArrayList<>(set);
        Collections.sort(ans);

        return ans;
    }

    /**
     * 2844. 生成特殊数字的最少操作
     * 给你一个下标从 0 开始的字符串 num ，表示一个非负整数。
     * 在一次操作中，您可以选择 num 的任意一位数字并将其删除。
     * 请注意，如果你删除 num 中的所有数字，则 num 变为 0。
     * 返回最少需要多少次操作可以使 num 变成特殊数字。
     * 如果整数 x 能被 25 整除，则该整数 x 被认为是特殊数字。
     * @param num
     * @return
     */
    public static int minimumOperations(String num) {
        int n = num.length();
        boolean have0 = false;
        boolean have5 = false;
        for (int i = n - 1; i >= 0; i--) {
            int c = num.charAt(i);
            if (have0 && (c == '0' || c == '5') ||
            have5 && (c == '2' || c == '7')) {
                return n-i-2;
            }
            if ( c == '0') {
                have0 = true;
            }
            if ( c == '5') {
                have5 = true;
            }
        }
        return have0 ? n-1 : n;
    }

    public static int findValueOfPartition(int[] nums) {
        Arrays.sort(nums);
        int ans = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length; i++) {
            ans = Math.min(ans,nums[i]-nums[i-1]);
        }
        return ans;
    }

    /**
     * 682. 棒球比赛
     * 你现在是一场采用特殊赛制棒球比赛的记录员。这场比赛由若干回合组成，
     * 过去几回合的得分可能会影响以后几回合的得分。
     * 比赛开始时，记录是空白的。你会得到一个记录操作的字符串列表 ops，
     * 其中 ops[i] 是你需要记录的第 i 项操作，ops 遵循下述规则：
     * 整数 x - 表示本回合新获得分数 x
     * "+" - 表示本回合新获得的得分是前两次得分的总和。题目数据保证记录此操作时前面总是存在两个有效的分数。
     * "D" - 表示本回合新获得的得分是前一次得分的两倍。题目数据保证记录此操作时前面总是存在一个有效的分数。
     * "C" - 表示前一次得分无效，将其从记录中移除。题目数据保证记录此操作时前面总是存在一个有效的分数。
     * 请你返回记录中所有得分的总和。
     * @param operations
     * @return
     */
    public static int calPoints(String[] operations) {
        int res = 0;
        List<Integer> list = new ArrayList<Integer>();
        for (String operation : operations) {
            int n = list.size();
            switch (operation.charAt(0)) {
                case '+':
                    res += list.get(n-1) + list.get(n-2);
                    list.add(list.get(n-1) + list.get(n-2));
                    break;
                case 'D':
                    res += 2 * list.get(n-1);
                    list.add(2 * list.get(n-1));
                    break;
                case 'C':
                    res -= list.get(n-1);
                    list.remove(n-1);
                    break;
                default:
                    res += Integer.parseInt(operation);
                    list.add(Integer.parseInt(operation));
                    break;
            }
        }
        return res;
    }

    /**
     * 2961. 双模幂运算
     * 给你一个下标从 0 开始的二维数组 variables ，其中
     * variables[i] = [ai, bi, ci, mi]，以及一个整数 target 。
     * 如果满足以下公式，则下标 i 是 好下标：
     * 0 <= i < variables.length
     * ((ai^bi % 10)^ci) % mi == target
     * 返回一个由好下标组成的数组，顺序不限 。
     * @param variables
     * @param target
     * @return
     */
    public static List<Integer> getGoodIndices(int[][] variables, int target) {
        List<Integer> ans = new ArrayList<Integer>();
        for (int i = 0; i<variables.length; i++) {
            int[] v = variables[i];
            if (powMod(powMod(v[0], v[1], 10), v[2], v[3]) == target) {
                ans.add(i);
            }
        }
        return ans;
    }

    //快速幂算法   X^4 = X^2 * X^2   X^3 = 1*X * X^2
    public static int powMod(int x, int y, int mod) {
        int res = 1;
        while (y != 0) {
            if ((y & 1) != 0) { //判断y是否为奇数
                res = res * x % mod; //奇数
            }
            x = x * x % mod;
            y >>= 1;
        }
        return res;
    }

    /**
     * 3111. 覆盖所有点的最少矩形数目
     * 给你一个二维整数数组 point ，其中 points[i] = [xi, yi]
     * 表示二维平面内的一个点。同时给你一个整数 w 。你需要用矩形 覆盖所有点。
     * 每个矩形的左下角在某个点 (x1, 0) 处，且右上角在某个点 (x2, y2) 处，
     * 其中 x1 <= x2 且 y2 >= 0 ，同时对于每个矩形都 必须 满足 x2 - x1 <= w。
     * 如果一个点在矩形内或者在边上，我们说这个点被矩形覆盖了。
     * 请你在确保每个点都 至少 被一个矩形覆盖的前提下，最少 需要多少个矩形。
     * 注意：一个点可以被多个矩形覆盖
     * @param points
     * @param w
     * @return
     */
    public static int minRectanglesToCoverPoints(int[][] points, int w) {
        Arrays.sort(points, (p,q) -> p[0] - q[0]);
        int ans = 0;
        int x2 = -1;
        for (int[] point : points) {
            if (point[0] > x2) {
                ans ++;
                x2 = point[0] + w;
            }
        }
        return ans;
    }

    /**
     * 3128. 直角三角形
     * 给你一个二维 boolean 矩阵 grid 。
     * 请你返回使用 grid 中的 3 个元素可以构建的 直角三角形 数目，且满足 3 个元素值 都 为 1 。
     * 注意：如果 grid 中 3 个元素满足：一个元素与另一个元素在 同一行，同时与第三个元素在 同一列 ，
     * 那么这 3 个元素称为一个 直角三角形 。这 3 个元素互相之间不需要相邻。
     * @param grid
     * @return
     */
    public static long numberOfRightTriangles(int[][] grid) {
        int n = grid[0].length;
        int[] colSum = new int[n];
        //将数组col中的所有元素都设置为-1。
        Arrays.fill(colSum, -1);
        for (int[] row: grid) {
            for (int j = 0; j < n; j++) {
                colSum[j] += row[j];
            }
        }
        long ans = 0;
        for (int[] row : grid) {
            int rowSum = -1;
            for (int x : row) {
                rowSum += x;
            }
            for (int j = 0; j < row.length; j++) {
                if (row[j] == 1) {
                    ans += rowSum * colSum[j];
                }
            }
        }
        return ans;
    }

    public static int findIntegers(int n) {
        int m = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        int[][] memo = new int[m][2];
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }
        return dfs(m-1, 0, true, n, memo);
    }

    public static int dfs(int i, int pre, boolean isLimit, int n, int[][] memo) {
        if (i < 0) {
            return 1;
        }
        if (!isLimit && memo[i][pre] >= 0) {
            return memo[i][pre];
        }
        int up = isLimit ? n >> i & 1 : 1;
        int res = dfs(i-1, 0, isLimit && up == 0, n, memo);
        if (pre == 0 && up == 1) {
            res += dfs(i-1, 1, isLimit, n, memo);
        }
        if (!isLimit) {
            memo[i][pre] = res;
        }
        return res;
    }

    public static void main(String[] args) {
        int n = 5;
        int a = findIntegers(n);
        System.out.println(a);
    }
}
