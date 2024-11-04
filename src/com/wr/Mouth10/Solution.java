package com.wr.Mouth10;

import com.wr.Mouth9.ListNode;
import com.wr.base.TreeNode;

import java.math.BigInteger;
import java.util.*;

/**
 * ClassName: Solution
 * Description:
 * date: 2024/10/13 15:11
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Solution {

    /**
     * 3162. 优质数对的总数 I
     * 给你两个整数数组nums1和nums2,长度分别为n和m.同时给你一个正整数k。
     * 如果nums1[i]可以被nums2[j]*k整除,则称数对(i,j)为优质数对
     * (0<=i<=n-1,0<=j<=m-1)。返回优质数对的总数。
     *
     * @param nums1
     * @param nums2
     * @param k
     * @return
     */
    public static int numberOfPairs(int[] nums1, int[] nums2, int k) {
        int sum = 0;
        for (int i = 0; i < nums1.length; i++) {
            if (nums1[i] % k != 0) continue;
            for (int j = 0; j < nums2.length; j++) {
                if (nums1[i] % (nums2[j] * k) == 0) {
                    sum++;
                }
            }
        }
        return sum;
    }

    /**
     * 2. 两数相加
     * 给你两个非空的链表,表示两个非负的整数.它们每位数字都是按照逆序的方式存储的,
     * 并且每个节点只能存储一位数字。请你将两个数相加，并以相同形式返回一个表示和的链表。
     * 你可以假设除了数字0之外，这两个数都不会以0开头。
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0); // 创建一个虚拟头节点
        ListNode current = dummyHead; // 当前节点指针
        int carry = 0; // 进位

        // 遍历两个链表
        while (l1 != null || l2 != null || carry > 0) {
            int sum = carry; // 从上一次的进位开始

            // 如果 l1 不为空，添加它的值
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next; // 移动到下一个节点
            }

            // 如果 l2 不为空，添加它的值
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next; // 移动到下一个节点
            }

            // 计算当前位的值和进位
            carry = sum / 10; // 更新进位
            current.next = new ListNode(sum % 10); // 创建新节点并添加到结果中
            current = current.next; // 移动到下一个节点
        }

        return dummyHead.next; // 返回结果链表（跳过虚拟头节点）
    }

    /**
     * 3164. 优质数对的总数 II
     * 给你两个整数数组nums1和nums2,长度分别为n和m.同时给你一个正整数k.
     * 如果nums1[i]可以被nums2[j]*k整除,则称数对(i,j)为优质数对
     * (0<=i<=n-1,0<=j<=m-1)。返回优质数对的总数。
     *
     * @param nums1
     * @param nums2
     * @param k
     * @return
     */
    public static long numberOfPairs1(int[] nums1, int[] nums2, int k) {
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : nums1) {
            if (x % k != 0) {
                continue;
            }
            x /= k;
            //寻找因子并更新计数
            for (int d = 1; d * d <= x; d++) {
                if (x % d > 0) {
                    continue;
                }
                cnt.merge(d, 1, Integer::sum);
                if (d * d < x) {
                    cnt.merge(x / d, 1, Integer::sum);
                }
            }
        }
        long ans = 0;
        for (int x : nums2) {
            ans += cnt.getOrDefault(x, 0);
        }
        return ans;
    }

    /**
     * 3158. 求出出现两次数字的XOR值
     * 给你一个数组nums,数组中的数字要么出现一次,要么出现两次.
     * 请你返回数组中所有出现两次数字的按位XOR值,如果没有数字出现过两次,返回0.
     *
     * @param nums
     * @return
     */
    public static int duplicateNumbersXOR(int[] nums) {
        int ans = 0;
        long vis = 0;
        for (int num : nums) {
            if ((vis >> num & 1) > 0) {
                ans ^= num;
            } else {
                vis |= 1L << num;
            }
        }
        return ans;
    }

    /**
     * 1884. 鸡蛋掉落-两枚鸡蛋
     * 给你2枚相同的鸡蛋，和一栋从第1层到第n层共有n层楼的建筑。
     * 已知存在楼层f,满足0<=f<=n,任何从高于f的楼层落下的鸡蛋都会碎,
     * 从f楼层或比它低的楼层落下的鸡蛋都不会碎.每次操作,你可以取一枚没有碎的鸡蛋
     * 并把它从任一楼层x扔下(满足1<=x<=n)。如果鸡蛋碎了,你就不能再次使用它.
     * 如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中重复使用这枚鸡蛋。
     * 请你计算并返回要确定f确切的值的最小操作次数是多少？
     *
     * @param n
     * @return
     */
    public int twoEggDrop(int n) {
        int[] memo = new int[1001];
        if (n == 0) {
            return 0;
        }
        if (memo[n] > 0) {
            return memo[n];
        }
        int res = Integer.MAX_VALUE;
        for (int i = 1; i <= n; i++) {
            res = Math.min(res, Math.max(i, twoEggDrop(n - i) + 1));
        }
        return memo[n] = res;
    }

    /**
     * 887. 鸡蛋掉落
     * 给你k枚相同的鸡蛋，并可以使用一栋从第1层到第n层共有n层楼的建筑。
     * 已知存在楼层f,满足0<=f<=n,任何从高于f的楼层落下的鸡蛋都会碎,从f楼层或比它低的楼层落下的鸡蛋都不会破。
     * 每次操作,你可以取一枚没有碎的鸡蛋并把它从任一楼层x扔下(满足1<=x<=n)。
     * 如果鸡蛋碎了,你就不能再次使用它.如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中
     * 重复使用这枚鸡蛋。请你计算并返回要确定f确切的值的最小操作次数是多少
     *
     * @param k
     * @param n
     * @return
     */

    public int superEggDrop(int k, int n) {
        Map<Integer, Integer> memo = new HashMap<Integer, Integer>();
        if (!memo.containsKey(n * 100 + k)) {
            int ans;
            if (n == 0) {
                ans = 0;
            } else if (k == 1) {
                ans = n;
            } else {
                int lo = 1, hi = n;
                while (lo + 1 < hi) {
                    int x = (lo + hi) / 2;
                    int t1 = superEggDrop(k - 1, x - 1);
                    int t2 = superEggDrop(k, n - x);

                    if (t1 < t2) {
                        lo = x;
                    } else if (t1 > t2) {
                        hi = x;
                    } else {
                        lo = hi = x;
                    }
                }
                ans = 1 + Math.min(Math.max(superEggDrop(k - 1, lo - 1), superEggDrop(k, n - lo)),
                        Math.max(superEggDrop(k - 1, hi - 1), superEggDrop(k, n - hi)));
            }
            memo.put(n * 100 + k, ans);
        }
        return memo.get(n * 100 + k);
    }

    /**
     * 3200. 三角形的最大高度
     * 给你两个整数red和blue，分别表示红色球和蓝色球的数量。你需要使用这些球来组成一个
     * 三角形,满足第1行有1个球,第2行有2个球,第3行有3个球,依此类推。
     * 每一行的球必须是相同颜色，且相邻行的颜色必须不同。返回可以实现的三角形的最大高度。
     * @param red
     * @param blue
     * @return
     */
    public int maxHeightOfTriangle(int red, int blue) {
        int[] cnt = new int[2];
        for (int i = 1; ; i++) {
            cnt[i%2] += i;
            if ((cnt[0] > red || cnt[1] > blue) && (cnt[0] > blue || cnt[1] > red)) {
                return i-1;
            }
        }
    }

    /**
     * 3194. 最小元素和最大元素的最小平均值
     * 你有一个初始为空的浮点数数组averages.另给你一个包含n个整数的数组nums，
     * 其中n为偶数。你需要重复以下步骤n/2次：
     * 从nums中移除最小 的元素 minElement 和 最大 的元素 maxElement。
     * 将(minElement+maxElement)/2加入到averages中。
     * 返回averages中的最小元素。
     * @param nums
     * @return
     */
    public static double minimumAverage(int[] nums) {
        Arrays.sort(nums);
        double sum = (double) (nums[0] + nums[nums.length-1] ) /2;
        for (int i = 0; i < nums.length/2; i++) {
            sum = Math.min(sum,(double) (nums[i] + nums[nums.length-1-i]) / 2);
        }
        return sum;
    }

    /**
     * 3191. 使二进制数组全部等于1的最少操作次数 I
     * 给你一个二进制数组 nums。你可以对数组执行以下操作任意次（也可以0次）：
     * 选择数组中任意连续3个元素,并将它们全部反转.反转一个元素指的是将它的值从0变1，
     * 或者从1变0。请你返回将nums中所有元素变为1的最少操作次数。
     * 如果无法全部变成1，返回-1。
     * @param nums
     * @return
     */
    public int minOperations(int[] nums) {
        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n - 2; i++) {
            if (nums[i] == 0) {
                nums[i+1] ^=1;
                nums[i+2] ^=1;
                ans++;
            }
        }
        return nums[n-2] != 0 && nums[n-1] != 0 ? ans : -1;
    }

    /**
     * 560. 和为 K 的子数组
     * 给你一个整数数组nums和一个整数k，请你统计并返回该数组中和为k的子数组的个数。
     * 子数组是数组中元素的连续非空序列。
     * @param nums
     * @param k
     * @return
     */
    public static int subarraySum(int[] nums, int k) {
//        int n = nums.length;
//        int[] preSum = new int[n+1];
//        for (int i = 0; i < n ; i++) {
//            preSum[i+1] = preSum[i] + nums[i];
//        }
//        int ans = 0;
//        for (int i = 1; i <= n; i++) {
//            for (int j = 0; j < i; j++) {
//                if (preSum[i] - preSum[j] == k) {
//                    ans++;
//                }
//            }
//        }
//        return ans;
        int n = nums.length;
        Map<Integer, Integer> preMap = new HashMap<>();
        preMap.put(0, 1);
        int preSum = 0;
        int res = 0;
        for(int i = 0; i < n; ++i){
            preSum += nums[i];
            res += preMap.getOrDefault(preSum - k, 0);
            preMap.put(preSum, preMap.getOrDefault(preSum, 0) + 1);
        }
        return res;
    }

    /**
     * 239. 滑动窗口最大值
     * 给你一个整数数组nums,有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。
     * 你只可以看到在滑动窗口内的k个数字.滑动窗口每次只向右移动一位。
     * 返回滑动窗口中的最大值。
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        return null;
    }

    /**
     * 910. 最小差值 II
     * 给你一个整数数组nums，和一个整数 k 。
     * 对于每个下标i（0 <= i < nums.length）,将nums[i]变成nums[i]+k
     * 或nums[i]-k。nums的分数是nums中最大元素和最小元素的差值。
     * 在更改每个下标对应的值之后，返回nums的最小分数 。
     * @param nums
     * @param k
     * @return
     */
    public static int smallestRangeII(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = nums[n-1] - nums[0];
        for (int i = 1; i < n; i++) {
            int mx = Math.max(nums[i-1] + k, nums[n-1]-k);
            int mn = Math.min(nums[0] + k, nums[i] - k);
            ans = Math.min(ans, mx-mn);
        }
        return ans;
    }

    /**
     * 3184. 构成整天的下标对数目 I
     * 给你一个整数数组hours,表示以小时为单位的时间,返回一个整数,表示满足i<j且
     * hours[i]+hours[j]构成整天的下标对i,j的数目。
     * 整天定义为时间持续时间是24小时的整数倍 。
     * 例如,1天是24小时,2天是48小时,3天是72小时,以此类推。
     * @param hours
     * @return
     */
    public static int countCompleteDayPairs1(int[] hours) {
        int ans = 0;
        for (int i = 0; i < hours.length; i++) {
            for (int j = i+1; j < hours.length; j++) {
                if ((hours[i] + hours[j]) % 24 == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }

    /**
     * 219. 存在重复元素 II
     * 给你一个整数数组nums和一个整数k，判断数组中是否存在两个不同的索引i和j，
     * 满足nums[i]==nums[j]且abs(i-j)<=k.如果存在，返回true;否则,返回false。
     * @param nums
     * @param k
     * @return
     */
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int length = nums.length;
        for (int i = 0; i < length; i++) {
            int num = nums[i];
            if (map.containsKey(num) && i - map.get(num) <= k) {
                return true;
            }
            map.put(num, i);
        }
        return false;
    }

    /**
     * 3185. 构成整天的下标对数目 II
     * 给你一个整数数组hours，表示以小时为单位的时间，返回一个整数，表示满足i<j
     * 且hours[i]+hours[j]构成整天的下标对i,j的数目。
     * 整天定义为时间持续时间是24小时的整数倍。
     * 例如,1天是24小时，2天是48小时，3天是72小时，以此类推。
     * @param hours
     * @return
     */
    public long countCompleteDayPairs(int[] hours) {
        final int H = 24;
        long ans = 0;
        int[] cnt = new int[H];
        for (int t : hours) {
            ans += cnt[(H - t%H) % H];
            cnt[t%H]++;
        }
        return ans;
    }

    /**
     * 3175.找到连续赢K场比赛的第一位玩家
     * 有n位玩家在进行比赛，玩家编号依次为0到n-1。
     * 给你一个长度为n的整数数组skills和一个正整数k，其中skills[i]是第i位玩家的技能
     * 等级。skills中所有整数互不相同。所有玩家从编号0到n-1排成一列。
     * 比赛进行方式如下：
     *     队列中最前面两名玩家进行一场比赛，技能等级更高的玩家胜出。
     *     比赛后，获胜者保持在队列的开头，而失败者排到队列的末尾。
     * 这个比赛的赢家是 第一位连续 赢下 k 场比赛的玩家。
     * 请你返回这个比赛的赢家编号。
     * @param skills
     * @param k
     * @return
     */
    public static int findWinningPlayer(int[] skills, int k) {
        int maxI = 0;
        int win = 0;
        for (int i = 1; i < skills.length && win < k; i++) {
            if (skills[i] > skills[maxI]) {
                maxI = i;
                win = 0;
            }
            win++;
        }
        return maxI;
    }

    /**
     * 11. 盛最多水的容器
     * 给定一个长度为n的整数数组height.有n条垂线,第i条线的两个端点是(i,0)和
     * (i,height[i])。找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。说明：你不能倾斜容器。
     * @param height
     * @return
     */
    public static int maxArea(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            int area = (right-left)*Math.min(height[left], height[right]);
            ans = Math.max(area, ans);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
     }

    /**
     * 3180. 执行操作可获得的最大总奖励 I
     * 给你一个整数数组rewardValues，长度为n，代表奖励的值。
     * 最初，你的总奖励x为0，所有下标都是未标记的。你可以执行以下操作任意次：
     * 从区间[0,n-1]中选择一个未标记的下标i。
     * 如果rewardValues[i]大于你当前的总奖励x,则将rewardValues[i]加到x上
     * （即 x = x + rewardValues[i]），并标记下标 i。
     * 以整数形式返回执行最优操作能够获得的最大总奖励。
     * @param rewardValues
     * @return
     */
    public static int maxTotalReward1(int[] rewardValues) {
        Arrays.sort(rewardValues);//排序
        int m = rewardValues[rewardValues.length - 1];//取最大值
        int[] dp = new int[2 * m];//设置dp函数
        dp[0] = 1;
        for (int x : rewardValues) {
            for (int k = 2 * x - 1; k >= x; k--) {
                if (dp[k-x] == 1) {
                    dp[k] = 1;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < dp.length; i++) {
            if (dp[i] == 1) {
                res = i;
            }
        }
        return res;
    }

    /**
     * 2915. 和为目标值的最长子序列的长度
     * 给你一个下标从0开始的整数数组nums和一个整数target。
     * 返回和为target的nums子序列中，子序列长度的最大值。如果不存在和为target的子序列，返回 -1 。
     * 子序列指的是从原数组中删除一些或者不删除任何元素后，剩余元素保持原来的顺序构成的数组。
     * @param nums
     * @param target
     * @return
     */
    public static int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        int[] f = new int[target + 1];
        Arrays.fill(f , Integer.MIN_VALUE);
        f[0] = 0;
        int s = 0;
        for (int x : nums) {
            s = Math.min(s + x, target);
            for (int j = s; j >= x; j--) {
                f[j] = Math.max(f[j], f[j-x] + 1);
            }
        }
        return f[target] > 0 ? f[target] : -1;
    }

    /**
     * 494. 目标和
     * 给你一个非负整数数组nums和一个整数target。
     * 向数组中的每个整数前添加'+'或'-'，然后串联起所有整数，可以构造一个表达式：
     * 例如,nums=[2,1]，可以在2之前添加'+'，在1之前添加'-'，
     * 然后串联起来得到表达式"+2-1"。
     * 返回可以通过上述方法构造的运算结果等于target的不同表达式的数目。
     * @param nums
     * @param target
     * @return
     */
    private static int[] nums;
    private static int[][] memo;
    public int findTargetSumWays(int[] nums, int target) {
        int s = 0;
        for (int x : nums) {
            s += x;
        }
        s-= Math.abs(target);
        if (s < 0 || s % 2 == 1) {
            return 0;
        }
        int m = s/2;
        Solution.nums = nums;
        int n = nums.length;
        memo = new int[n][m+1];
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }
        return dfs(n-1, m);
    }

    private static int dfs(int i, int c) {
        if (i < 0) {
            return c == 0?1:0;
        }
        if (memo[i][c] != -1) {
            return memo[i][c];
        }
        if (c < nums[i]) {
            return memo[i][c] = dfs(i - 1, c);
        }
        return memo[i][c] = dfs(i-1, c) + dfs(i-1, c-nums[i]);
    }

    /**
     * 3181. 执行操作可获得的最大总奖励 II
     * 给你一个整数数组rewardValues，长度为n，代表奖励的值。
     * 最初，你的总奖励x为0，所有下标都是未标记的。你可以执行以下操作任意次 ：
     * 从区间 [0, n - 1] 中选择一个未标记的下标 i。
     * 如果rewardValues[i]大于你当前的总奖励x,则将rewardValues[i]加到x上
     * （即 x = x + rewardValues[i]），并标记下标i。
     * 以整数形式返回执行最优操作能够获得的最大总奖励。
     * @param rewardValues
     * @return
     */
//    public int maxTotalReward(int[] rewardValues) {
//        int m = 0;
//        for (int v : rewardValues) {
//            m = Math.max(m, v);
//        }
//        for (int v : rewardValues) {
//            if (v == m-1) {
//                return m * 2 - 1;
//            }
//        }
//        BigInteger f = BigInteger.ONE;
//        for (int v : Arrays.stream(rewardValues).distinct().sorted().toArray()) {
//            BigInteger mask = BigInteger.ONE.shiftLeft(v).subtract(BigInteger.ONE);
//            f = f.or(f.and(mask).shiftLeft(v));
//        }
//        return f.bitLength()-1;
//    }
    public int maxTotalReward(int[] rewardValues, int index, int sum, Map<Integer,Integer> memo){
        int key = (index << 15) | sum;

        if(memo.containsKey(key)){
            return memo.get(key);
        }

        if(sum <= 1 || index < 0){
            return 0;
        }
        int res = 0;


        int start = 0;
        int end = index;
        while(start <= end){
            int mid = (start + end) / 2;
            if(rewardValues[mid] >= sum){
                end = mid - 1;
            }else{
                start = mid + 1;
            }
        }

        for(int i = end;i >= 0;i--){
            int up = Math.min(sum - rewardValues[i], rewardValues[i]);
            int leftRes = maxTotalReward(rewardValues, i - 1, up, memo);
            res = Math.max(res, rewardValues[i] + leftRes);
            if(res == sum - 1){
                break;
            }
        }
        memo.put(key, res);
        return res;
    }
    public int maxTotalReward(int[] rewardValues) {
        Arrays.sort(rewardValues);
        int max = rewardValues[rewardValues.length - 1];
        return maxTotalReward(rewardValues, rewardValues.length - 2, max,new HashMap<>()) + max;
    }

    /**
     * 684. 冗余连接
     * 树可以看成是一个连通且无环的无向图。
     * 给定往一棵n个节点(节点值1～n)的树中添加一条边后的图。添加的边的两个顶点包含在
     * 1到n中间,且这条附加的边不属于树中已存在的边。图的信息记录于长度为n的二维数组
     * edges，edges[i]=[ai,bi]表示图中在ai和bi之间存在一条边。
     * 请找出一条可以删去的边，删除后可使得剩余部分是一个有着n个节点的树。
     * 如果有多个答案，则返回数组edges中最后出现的那个。
     * @param edges
     * @return
     */
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] parent = new int[n+1];
        for (int i = 1; i <= n; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < n; i++) {
            int[] edge = edges[i];
            int node1 = edge[0], node2 = edge[1];
            if (find(parent, node1) != find(parent, node2)) {
                union(parent, node1, node2);
            } else {
                return edge;
            }
        }
        return new int[0];
    }

    public void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    public int find(int[] parent, int index) {
        if (parent[index] != index) {
            parent[index] = find(parent, parent[index]);
        }
        return parent[index];
    }

    /**
     * 53. 最大子数组和
     * 给你一个整数数组nums,请你找出一个具有最大和的连续子数组(子数组最少包含一个
     * 元素），返回其最大和。子数组是数组中的一个连续部分。
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int ans = Integer.MIN_VALUE;
        int minPreSum = 0;
        int preSum = 0;
        for (int x : nums) {
            preSum += x;
            ans = Math.max(ans, preSum - minPreSum);
            minPreSum = Math.min(minPreSum, preSum);
        }
        return ans;
    }

    /**
     * 17. 电话号码的字母组合   回溯
     * 给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按
     * 任意顺序返回.给出数字到字母的映射如下(与电话按键相同).注意1不对应任何字母。
     * @param digits
     * @return
     */
    private static final String[] MAPPING = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    private static final List<String> ans = new ArrayList<>();
    private static char[] digits;
    private static char[] path;
    public List<String> letterCombinations(String digits) {
        int n = digits.length();
        if (n == 0) {
            return List.of();
        }
        Solution.digits = digits.toCharArray();
        path = new char[n];
        dfs11(0);
        return ans;
    }

    private static void dfs11(int i) {
        if (i == digits.length) {
            ans.add(new String(path));
            return;
        }
        for (char c : MAPPING[digits[i] - '0'].toCharArray()) {
            path[i] = c;
            dfs11(i+1);
        }
    }

    /**
     * 78. 子集
     * 给你一个整数数组nums，数组中的元素互不相同.返回该数组所有可能的子集(幂集).
     * 解集不能包含重复的子集。你可以按任意顺序返回解集。
     * @param nums
     * @return
     */
    private static final List<List<Integer>> ans1 = new ArrayList<>();
    private static final List<Integer> path1 = new ArrayList<>();
    private static int[] nums1;
    public List<List<Integer>> subsets(int[] nums) {
        nums1 = nums;
        dfs12(0);
        return ans1;
    }

    private static void dfs12(int i) {
        if (i == nums1.length) {
            ans1.add(new ArrayList<>(path1));
            return;
        }
        dfs12(i+1);

        path1.add(nums1[i]);
        dfs12(i+1);
        path1.remove(path1.size() - 1);
    }

    /**
     * 3211. 生成不含相邻零的二进制字符串
     * 给你一个正整数n。如果一个二进制字符串x的所有长度为2的子字符串
     * 中包含至少一个"1"，则称x是一个有效字符串。
     * 返回所有长度为n的有效字符串，可以以任意顺序排列。
     * @param n
     * @return
     */
    public List<String> validStrings(int n) {
        List<String> ans = new ArrayList<>();
        char[] path = new char[n];
        dfs13(0, n, path, ans);
        return ans;
    }

    private static void dfs13(int i, int n, char[] path, List<String> ans) {
        if (i == n) {
            ans.add(new String(path));
            return;
        }
        path[i] = '1';
        dfs13(i+1, n, path, ans);

        if (i == 0 || path[i-1] == '1') {
            path[i] = '0';
            dfs13(i+1, n, path, ans);
        }
    }

    /**
     * 257. 二叉树的所有路径
     * 给你一个二叉树的根节点root,按任意顺序,返回所有从根节点到叶子节点的路径。
     * 叶子节点是指没有子节点的节点。
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> ans = new ArrayList<>();
        dfs2(root, "", ans);
        return ans;
    }

    private static void dfs2(TreeNode root, String path, List<String> paths) {
        if (root == null) {
            return;
        }
        StringBuffer path1 = new StringBuffer(path);
        path1.append(Integer.toString(root.val));
        if (root.left == null && root.right == null) {
            paths.add(path1.toString());
        } else {
            path1.append("->");
            dfs2(root.left, path1.toString(), paths);
            dfs2(root.right, path1.toString(), paths);
        }
    }

    /**
     * 3216. 交换后字典序最小的字符串
     * 给你一个仅由数字组成的字符串s，在最多交换一次相邻且具有相同奇偶性的数字后,
     * 返回可以得到的字典序最小的字符串.如果两个数字都是奇数或都是偶数，
     * 则它们具有相同的奇偶性。例如，5和9、2和4奇偶性相同，而6和9奇偶性不同。
     * @param s
     * @return
     */
    public static String getSmallestString(String s) {
        char[] charArray = s.toCharArray();
        for (int i = 1; i < charArray.length; i++) {
            int a = charArray[i];
            int j = i-1;
            int b = charArray[j];
            if (a < b && a % 2 == b % 2) {
                charArray[i] = (char) b;
                charArray[j] = (char) a;
                break;
            }
        }
        return new String(charArray);
    }

    /**
     * 189. 轮转数组
     * 给定一个整数数组nums，将数组中的元素向右轮转k个位置,其中k是非负数。
     * @param nums
     * @param k
     */
    public static void rotate(int[] nums, int k) {
        int n = nums.length;
        k %= n;
        reverse(nums, 0, n-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, n-1);
    }

    private static void reverse(int[] nums, int i, int j) {
        while (i < j) {
            int temp = nums[i];
            nums[i++] = nums[j];
            nums[j--] = temp;
        }
    }

    /**
     * 13. 罗马数字转整数
     * 罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。
     * 字符          数值
     * I             1
     * V             5
     * X             10
     * L             50
     * C             100
     * D             500
     * M             1000
     * 例如,罗马数字2写做II,即为两个并列的1。12写做XII，即为X+II。
     * 27写做XXVII,即为XX+V+II。
     * 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，
     * 例如4不写做IIII,而是IV。数字1在数字5的左边,所表示的数等于大数5减小数1
     * 得到的数值4。同样地,数字9表示为IX.这个特殊的规则只适用于以下六种情况：
     *     I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
     *     X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
     *     C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
     * 给定一个罗马数字，将其转换成整数。
     * @param s
     * @return
     */
    public static int romanToInt(String s) {
        Map map = new HashMap();
        int ans = 0;
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        char[] c = s.toCharArray();
        for (int i = 0; i < c.length; i++) {
            if ('I'== c[i]) {
                if (i < c.length-1 && 'V' == c[i+1]) {
                    ans += 4;
                    i++;
                    continue;
                }
                if (i < c.length-1 && 'X' == c[i+1]) {
                    ans += 9;
                    i++;
                    continue;
                }
            } if ('X' == c[i]) {
                if (i < c.length-1 && 'L' == c[i+1]) {
                    ans += 40;
                    i++;
                    continue;
                }
                if (i < c.length-1 && 'C' == c[i+1]) {
                    ans += 90;
                    i++;
                    continue;
                }
            }
            if ('C' == c[i]) {
                if (i < c.length-1 && 'D' == c[i + 1]) {
                    ans += 400;
                    i++;
                    continue;
                }
                if (i < c.length-1 && 'M' == c[i+1]) {
                    ans += 900;
                    i++;
                    continue;
                }
            }
            ans += (int)map.get(c[i]);
        }
        return ans;
    }

    /**
     * 55. 跳跃游戏
     * 给你一个非负整数数组nums,你最初位于数组的第一个下标.数组中的每个元素代表
     * 你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标,
     * 如果可以,返回true;否则,返回false。
     * @param nums
     * @return
     */
    public static boolean canJump(int[] nums) {
//        int mx = 0;
//        for (int i = 0; i < nums.length; i++) {
//            if (i > mx) {
//                return false;
//            }
//            mx = Math.max(mx, i+nums[i]);
//        }
//        return true;
        int jump = nums.length - 1;
        for(int i = nums.length - 1; i >= 0; i--){
            if(jump <= i + nums[i]) jump = i;
        }
        return jump == 0;
    }

    /**
     * 3165. 不包含相邻元素的子序列的最大和
     * 给你一个整数数组nums和一个二维数组queries,其中queries[i]=[posi,xi]。
     * 对于每个查询i,首先将nums[posi]设置为xi,然后计算查询i的答案,该答案为nums中
     * 不包含相邻元素的子序列的最大和。返回所有查询的答案之和。
     * 由于最终答案可能非常大，返回其对109+7取余的结果。
     * 子序列是指从另一个数组中删除一些或不删除元素而不改变剩余元素顺序得到的数组。
     * @param nums
     * @param queries
     * @return
     */
    public int maximumSumSubsequence(int[] nums, int[][] queries) {
        int n = nums.length;
        long[][] t = new long[2 << (32 - Integer.numberOfLeadingZeros(n))][4];
        build(t, nums, 1, 0, n-1);

        long ans = 0;
        for (int[] q : queries) {
            update(t, 1, 0, n - 1, q[0], q[1]);
            ans += t[1][3];
        }
        return (int) (ans % 1_000_000_007);

    }

    private void maintain(long[][] t, int o) {
        long[] a = t[o * 2];
        long[] b = t[o * 2 + 1];
        t[o][0] = Math.max(a[0] + b[2], a[1] + b[0]);
        t[o][1] = Math.max(a[0] + b[3], a[1] + b[1]);
        t[o][2] = Math.max(a[2] + b[2], a[3] + b[0]);
        t[o][3] = Math.max(a[2] + b[3], a[3] + b[1]);
    }

    private void build(long[][] t, int[] nums, int o, int l, int r) {
        if (l == r) {
            t[o][3] = Math.max(nums[l], 0);
            return;
        }
        int m = (l + r) / 2;
        build(t, nums, o * 2, l, m);
        build(t, nums, o * 2 + 1, m + 1, r);
        maintain(t, o);
    }

    private void update(long[][] t, int o, int l, int r, int i, int val) {
        if (l == r) {
            t[o][3] = Math.max(val, 0);
            return;
        }
        int m = (l + r) / 2;
        if (i <= m) {
            update(t, o * 2, l, m, i, val);
        } else {
            update(t, o * 2 + 1, m + 1, r, i, val);
        }
        maintain(t, o);
    }

    /**
     * 45. 跳跃游戏 II
     * 给定一个长度为n的0索引整数数组nums。初始位置为nums[0]。
     * 每个元素nums[i]表示从索引i向前跳转的最大长度。换句话说，如果你在nums[i]处，
     * 你可以跳转到任意nums[i+j]处: 0 <= j <= nums[i],i + j < n
     * 返回到达nums[n-1]的最小跳跃次数。生成的测试用例可以到达nums[n-1]。
     * @param nums
     * @return
     */
    public static int jump(int[] nums) {
        int ans = 0;
        int curRight = 0;
        int nextRight = 0;
        for (int i = 0; i < nums.length-1; i++) {
            nextRight = Math.max(nextRight, i+nums[i]);
            if (i == curRight) {
                curRight = nextRight;
                ans++;
            }
        }
        return ans;
    }

    /**
     * 58. 最后一个单词的长度
     * 给你一个字符串s,由若干单词组成,单词前后用一些空格字符隔开.
     * 返回字符串中最后一个单词的长度。
     * 单词是指仅由字母组成、不包含任何空格字符的最大子字符串。
     * @param s
     * @return
     */
    public static int lengthOfLastWord(String s) {
        char[] c = s.toCharArray();
        int n = c.length;
        int ans = 0;
        for (int i = n-1; i >= 0; i--) {
            if (c[i] != ' ') {
                ans++;
                continue;
            }
            if (ans !=0 && c[i] == ' '){
                break;
            }
        }
        return ans;
    }

    /**
     * 14. 最长公共前缀
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     * 如果不存在公共前缀，返回空字符串 ""。
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {
        StringBuilder stringBuilder = new StringBuilder();
        String shortest = strs[0]; // 假设第一个字符串是最短的

        // 遍历数组，找出最短的字符串
        for (String str : strs) {
            if (str.length() < shortest.length()) {
                shortest = str; // 更新最短字符串
            }
        }
        int s = 0;
        for (int i = 0; i < shortest.length(); i++) {
            char c1 = strs[0].charAt(i);
            int k = 1;
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].charAt(i) == c1) {
                    k++;
                }
            }
            if (k==strs.length) {
                stringBuilder.append(c1);
            }
            else {
                break;
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 151. 反转字符串中的单词
     * 给你一个字符串s，请你反转字符串中单词的顺序。
     * 单词是由非空格字符组成的字符串。s中使用至少一个空格将字符串中的单词分隔开。
     * 返回单词顺序颠倒且单词之间用单个空格连接的结果字符串。
     * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。
     * 返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
     * @param s
     * @return
     */
    public static String reverseWords(String s) {
        int n = s.length();
        StringBuilder stringBuilder = new StringBuilder();
        int flag = 0;
        for (int i = n-1; i >= 0; i--) {
            if (s.charAt(i) != ' ') {
                flag = 1;
                stringBuilder.append(s.charAt(i));
            }
            if (flag == 1 && s.charAt(i) == ' ') {
                stringBuilder.append(' ');
                flag = 0;
            }
        }
        String input = stringBuilder.toString();

        // 按空格分割字符串
        String[] words = input.split(" ");

        // 反转每个单词
        StringBuilder result = new StringBuilder();
        for (String word : words) {
            StringBuilder reversedWord = new StringBuilder(word).reverse();
            result.append(reversedWord).append(" ");
        }
        return result.toString().trim();
    }

    /**
     * 3226. 使两个整数相等的位更改次数
     * 给你两个正整数n和k,你可以选择n的二进制表示中任意一个值为1的位，并将其改为0。
     * 返回使得n等于k所需要的更改次数。如果无法实现，返回-1。
     * @param n
     * @param k
     * @return
     */
    public static int minChanges(int n, int k) {
//        String n1 = String.format("%1024s", Integer.toBinaryString(n)).replace(' ', '0');;
//        String k1 = String.format("%1024s", Integer.toBinaryString(k)).replace(' ', '0');;
//        int ans = 0;
//        int l = Math.min(n1.length(), k1.length());
//        for (int i = 0; i < l; i++) {
//            if (n1.charAt(i) != k1.charAt(i)) {
//                if (n1.charAt(i) == '0') {
//                    return -1;
//                }
//                ans ++;
//            }
//        }
//        return ans;
        return (n & k) != k ? -1 : Integer.bitCount(n ^ k);
    }

    /**
     * 125. 验证回文串
     * 如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，
     * 短语正着读和反着读都一样。则可以认为该短语是一个回文串 。
     * 字母和数字都属于字母数字字符.给你一个字符串s,如果它是回文串,返回true;
     * 否则,返回 false。
     * @param s
     * @return
     */
    public static boolean isPalindrome(String s) {
        String str = s.toLowerCase();

        String s1 = str.replaceAll("[^a-zA-Z0-9]", "");
        int left = 0;
        int right = s1.length()-1;
        while (left < right) {
            if (s1.charAt(left) != s1.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    public static void main(String[] args) {
        int[] nums = {2,3,0,1,4};
        int n = 44, k = 2;
        int c = 5;
        String[] strs = {"flower","flow","flight"};
        String  s = "A man, a plan, a canal: Panama";
//        System.out.println(judgeSquareSum(c));
    }
}