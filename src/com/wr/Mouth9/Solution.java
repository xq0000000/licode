package com.wr.Mouth9;


import com.wr.base.TreeNode;

import java.util.*;

/**
 * ClassName: Solution
 * Description:
 * date: 2024/9/3 14:56
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Solution {
    /**
     * 2708. 一个小组的最大实力值
     * 给你一个下标从 0 开始的整数数组 nums ，它表示一个班级中所有学生
     * 在一次考试中的成绩。老师想选出一部分同学组成一个非空小组，且这个小组的
     * 实力值最大，如果这个小组里的学生下标为 i0,i1,i2,...,ik，那么这个小组
     * 的实力值定义为 nums[i0] * nums[i1] * nums[i2] * ... * nums[ik]。
     * 请你返回老师创建的小组能得到的最大实力值为多少。
     *
     * @param nums
     * @return
     */
    public static long maxStrength(int[] nums) {
        int n = nums.length;
        int[] a = new int[n];
        long sum = 1;
        int k = 0;
        boolean flag = false;
        if (nums[0] < 0 && nums.length == 1) {
            return nums[0];
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] < 0) {
                a[k++] = nums[i];
            }
            if (nums[i] != 0) {
                sum *= nums[i];
            }
            if (nums[i] > 0) {
                flag = true;
            }
        }
        if (!flag) {
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (nums[i] < 0) {
                    m++;
                }
            }
            if (m <= 1) {
                return 0;
            }
        }
        if (k % 2 == 0) {
            return sum;
        } else {
            long max = toMax(a);
            return sum / max;
        }
    }

    public static long toMax(int[] a) {
        long max = a[0];
        for (int i : a) {
            if (i > max && i != 0) {
                max = i;
            }
        }
        return max;
    }

    /**
     * 2469. 温度转换
     * 给你一个四舍五入到两位小数的非负浮点数 celsius 来表示温度，以摄氏度
     * （Celsius）为单位。你需要将摄氏度转换为 开氏度（Kelvin）和 华氏度
     * （Fahrenheit），并以数组 ans = [kelvin, fahrenheit] 的形式
     * 返回结果。返回数组ans.与实际答案误差不超过 10-5 的会视为正确答案
     * 开氏度 = 摄氏度 + 273.15
     * 华氏度 = 摄氏度 * 1.80 + 32.00
     *
     * @param celsius
     * @return
     */
    public static double[] convertTemperature(double celsius) {
        double[] d = new double[2];
        double Kelvin = celsius + 273.15;
        double fahrenheit = (celsius * 1.80) + 32.00;
        d[0] = Kelvin;
        d[1] = fahrenheit;
        return d;
    }

    /**
     * 2413. 最小偶倍数
     * 给你一个正整数 n ，返回 2 和 n 的最小公倍数（正整数）。
     *
     * @param n
     * @return
     */
    public static int smallestEvenMultiple(int n) {
        if (n % 2 == 0) {
            return n;
        } else {
            return n * 2;
        }
    }

    /**
     * 1486. 数组异或操作
     * 给你两个整数，n 和 start 。
     * 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且
     * n == nums.length。
     * 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
     *
     * @param n
     * @param start
     * @return
     */
    public static int xorOperation(int n, int start) {
        int[] nums = new int[n];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum ^= start + 2 * i;
        }
        return sum;
    }

    /**
     * 1512. 好数对的数目
     * 给你一个整数数组 nums 。
     * 如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，
     * 就可以认为这是一组 好数对 。
     * 返回好数对的数目。
     *
     * @param nums
     * @return
     */
    public static int numIdenticalPairs(int[] nums) {
        int n = nums.length;
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] == nums[j]) {
                    k++;
                }
            }
        }
        return k;
    }

    /**
     * 1534. 统计好三元组
     * 给你一个整数数组arr，以及a、b 、c三个整数。请你统计其中好三元组的数量。
     * 如果三元组 (arr[i],arr[j],arr[k])满足下列全部条件，则认为它是一个
     * 好三元组。0 <= i < j < k < arr.length
     * |arr[i] - arr[j]| <= a
     * |arr[j] - arr[k]| <= b
     * |arr[i] - arr[k]| <= c
     * 其中 |x| 表示 x 的绝对值。
     * 返回 好三元组的数量 。
     *
     * @return
     */
    public static int countGoodTriplets(int[] arr, int a, int b, int c) {
        int m = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                for (int k = j + 1; k < arr.length; k++) {
                    if ((Math.abs(arr[i] - arr[j]) <= a)
                            && (Math.abs(arr[j] - arr[k]) <= b)
                            && (Math.abs(arr[i] - arr[k]) <= c)) {
                        m++;
                    }
                }
            }
        }
        return m;
    }

    /**
     * 709. 转换成小写字母
     * 给你一个字符串 s ，将该字符串中的大写字母转换成相同的小写字母，
     * 返回新的字符串。
     *
     * @param s
     * @return
     */
    public static String toLowerCase(String s) {
        StringBuilder sb = new StringBuilder();
        char[] c = new char[s.length()];
        for (int i = 0; i < s.length(); i++) {
            sb.append(Character.toLowerCase(s.charAt(i)));
        }
        return sb.toString();
    }

    /**
     * 258. 各位相加
     * 给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。
     * 返回这个结果。
     *
     * @param num
     * @return
     */
    public static int addDigits(int num) {
        while (num >= 10) {
            int sum = 0;
            while (num > 0) {
                sum += num % 10;
                num /= 10;
            }
            num = sum;
        }
        return num;
    }

    /**
     * 231. 2 的幂
     * 给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；
     * 否则，返回 false 。如果存在一个整数 x 使得 n==2x,则认为 n是2的幂次方。
     *
     * @param n
     * @return
     */
    public static boolean isPowerOfTwo(int n) {
        int i = 0;
        for (i = 0; i < n; i++) {
            if (Math.pow(2, i) == n) {
                break;
            }
        }
        if (i < n) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * 326. 3 的幂
     * 给定一个整数,写一个函数来判断它是否是3的幂次方。如果是,返回 true;
     * 否则，返回 false 。整数 n是3的幂次方需满足：存在整数x使得n==3x
     *
     * @param n
     * @return
     */
    public static boolean isPowerOfThree(int n) {
        while (n != 0 && n % 3 == 0) {
            n /= 3;
        }
        return n == 1;
    }

    /**
     * 263. 丑数
     * 丑数 就是只包含质因数 2、3 和 5 的正整数。
     * 给你一个整数 n ，请你判断 n 是否为 丑数 。
     * 如果是，返回 true ；否则，返回 false 。
     *
     * @param n
     * @return
     */
    public static boolean isUgly(int n) {
        int[] a = {2, 3, 5};
        for (int i = 0; i < 3; i++) {
            while (n != 0 && n % a[i] == 0) {
                n /= a[i];
            }
        }
        return n == 1;
    }

    /**
     * 1470. 重新排列数组
     * 给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
     * 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。
     *
     * @param nums
     * @param n
     * @return
     */
    public static int[] shuffle(int[] nums, int n) {
        int[] a = new int[2 * n];
        for (int i = 0; i < n; i++) {
            a[2 * i] = nums[i];
            a[2 * i + 1] = nums[i + n];
        }
        return a;
    }

    /**
     * 2860. 让所有学生保持开心的分组方法数
     * 给你一个下标从0开始、长度为n的整数数组nums，其中n是班级中学生的总数。
     * 班主任希望能够在让所有学生保持开心的情况下选出一组学生：
     * 如果能够满足下述两个条件之一，则认为第 i 位学生将会保持开心：
     * 这位学生被选中，并且被选中的学生人数 严格大于 nums[i] 。
     * 这位学生没有被选中，并且被选中的学生人数 严格小于 nums[i] 。
     * 返回能够满足让所有学生保持开心的分组方法的数目。
     *
     * @param nums
     * @return
     */
    public int countWays(List<Integer> nums) {
        int[] a = nums.stream().mapToInt(i -> i).toArray();
        Arrays.sort(a);
        int ans = a[0] > 0 ? 1 : 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i - 1] < i && i < a[i]) {
                ans++;
            }
        }
        return ans + 1;
    }

    public static void sum(double[] a, double b) {
        double sum = 0;
        for (double v : a) {
            sum += v;
        }
        System.out.println(sum);
        System.out.println(sum - b);
    }

    /**
     * 3174. 清除数字
     * 给你一个字符串 s 。
     * 你的任务是重复以下操作删除所有数字字符：
     * 删除第一个数字字符以及它左边最近的非数字字符。
     * 请你返回删除所有数字字符以后剩下的字符串。
     *
     * @param s
     * @return
     */
    public String clearDigits(String s) {
        StringBuilder st = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                st.deleteCharAt(st.length() - 1);
            } else {
                st.append(c);
            }
        }
        return st.toString();
    }

    /**
     * 3176. 求出最长好子序列 I
     * 给你一个整数数组 nums 和一个 非负 整数 k 。如果一个整数序列 seq 满足在
     * 范围下标范围 [0, seq.length - 2] 中存在 不超过 k 个下标 i 满足
     * seq[i] != seq[i + 1] ，那么我们称这个整数序列为 好 序列。
     * 请你返回 nums 中 好子序列的最长长度
     *
     * @param nums
     * @param k
     * @return
     */
    public int maximumLength(int[] nums, int k) {
        return 0;
    }

    /**
     * 2181. 合并零之间的节点
     * 给你一个链表的头节点head,该链表包含由0分隔开的一连串整数。链表的开端和末尾
     * 的节点都满足 Node.val == 0 。对于每两个相邻的0,请你将它们之间的所有节点
     * 合并成一个节点，其值是所有已合并节点的值之和。然后将所有0移除，
     * 修改后的链表不应该含有任何 0. 返回修改后链表的头节点 head 。
     *
     * @param head
     * @return
     */
    public static ListNode mergeNodes(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = new ListNode(0);
        ListNode current = node;
        while (head.next != null) {
            if (head.val == 0) {
                int sum = 0;
                head = head.next;
                while (head.val != 0) {
                    sum += head.val;
                    head = head.next;
                }
                current.next = new ListNode(sum);
                current = current.next;
            }
        }
        return node.next;
    }

    /**
     * 1290. 二进制链表转整数
     * 给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。
     * 已知此链表是一个整数数字的二进制表示形式。
     * 请你返回该链表所表示数字的 十进制值
     *
     * @param head
     * @return
     */
    public static int getDecimalValue(ListNode head) {
        int sum = 0;
        while (head != null) {
            sum = head.val + sum * 2;
            head = head.next;
        }
        return sum;
    }

    /**
     * 203. 移除链表元素
     * 给你一个链表的头节点head和一个整数 val,请你删除链表中所有满足
     * Node.val==val的节点，并返回新的头节点。
     *
     * @param head
     * @param val
     * @return
     */
    public static ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0, head);
        ListNode current = dummy;
        while (current.next != null) {
            if (current.next.val == val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return dummy.next;
    }

    /**
     * 3217. 从链表中移除在数组中存在的节点
     * 给你一个整数数组 nums 和一个链表的头节点 head。从链表中移除所有存在于
     * nums 中的节点后，返回修改后的链表的头节点。
     *
     * @param nums
     * @param head
     * @return
     */
    public static ListNode modifiedList(int[] nums, ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode current = dummy;
        Set<Integer> set = new HashSet<>(nums.length);
        for (int num : nums) {
            set.add(num);
        }
        while (current.next != null) {
            int val = current.next.val;
            if (set.contains(val)) { //直接for循环会超时，用set可以减少时间
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return dummy.next;
    }

    /**
     * 83. 删除排序链表中的重复元素
     * 给定一个已排序的链表的头head,删除所有重复的元素,使每个元素只出现一次.
     * 返回已排序的链表.
     *
     * @param head
     * @return
     */
    public static ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode current = dummy;
        while (current.next != null && current.next.next != null) {
            if (current.next.val == current.next.next.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return dummy.next;
    }

    /**
     * 206. 反转链表
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     *
     * @param head
     * @return
     */
    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }

    /**
     * 19. 删除链表的倒数第 N 个结点
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     *
     * @param head
     * @param n
     * @return
     */
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p1 = dummy;
        while (n > 0) {
            p1 = p1.next;
            n--;
        }
        ListNode p2 = dummy;
        while (p1.next != null) {
            p1 = p1.next;
            p2 = p2.next;
        }
        p2.next = p2.next.next;
        return dummy.next;
    }

    public static ListNode convertArrayToList(int[] array) {
        if (array == null || array.length == 0) {
            return null;
        }
        ListNode head = new ListNode(0);
        ListNode current = head;
        for (int value : array) {
            current.next = new ListNode(value);
            current = current.next;
        }
        return head.next;
    }

    public static void printList(ListNode node) {
        while (node != null) {
            System.out.print(node.val + " -> ");
            node = node.next;
        }
        System.out.println("null");
    }

    /**
     * 2552. 统计上升四元组
     * 给你一个长度为n下标从0开始的整数数组 nums,它包含1到n的所有数字，
     * 请你返回上升四元组的数目。如果一个四元组 (i, j, k, l) 满足以下条件，
     * 我们称它是上升的：0 <= i < j < k < l < n 且
     * nums[i] < nums[k] < nums[j] < nums[l] 。
     *
     * @param nums
     * @return
     */
    public static long countQuadruplets(int[] nums) {
        long cnt4 = 0;
        long[] cnt3 = new long[nums.length];
        for (int l = 2; l < nums.length; l++) {
            int cnt2 = 0;
            for (int j = 0; j < l; j++) {
                if (nums[j] < nums[l]) {
                    cnt4 += cnt3[j];
                    cnt2++;
                } else {
                    cnt3[j] += cnt2;
                }
            }
        }
        return cnt4;
    }

    /**
     * 141. 环形链表
     * 给你一个链表的头节点head,判断链表中是否有环。
     * 如果链表中有某个节点,可以通过连续跟踪next指针再次到达,
     * 则链表中存在环.为了表示给定链表中的环,评测系统内部使用整数pos
     * 来表示链表尾连接到链表中的位置(索引从0开始)。注意:pos不作为参数进行传递。
     * 仅仅是为了标识链表的实际情况。如果链表中存在环,则返回 true。否则,返回 false
     *
     * @param head
     * @return
     */
    public static boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode l1 = head.next;
        ListNode l2 = head.next;
        while (l2 != null && l2.next != null) {
            l1 = l1.next;
            l2 = l2.next.next;
            if (l1 == l2) {
                return true;
            }
        }
        return false;
    }

    /**
     * 2555. 两个线段获得的最多奖品
     * 在X轴上有一些奖品。给你一个整数数组prizePositions,它按照非递减顺序排列，
     * 其中prizePositions[i]是第i件奖品的位置.数轴上一个位置可能会有多件奖品
     * 再给你一个整数 k 。你可以选择两个端点为整数的线段。每个线段的长度都必须是k。
     * 你可以获得位置在任一线段上的所有奖品(包括线段的两个端点)。注意,两个线段可能会有相交。
     * 比方说k=2,你可以选择线段[1,3]和[2,4],你可以获得满足
     * 1<=prizePositions[i]<=3或者2<=prizePositions[i]<=4的所有奖品i。
     * 请你返回在选择两个最优线段的前提下，可以获得的最多奖品数目。
     *
     * @param prizePositions
     * @param k
     * @return
     */
    public static int maximizeWin(int[] prizePositions, int k) {
        int n = prizePositions.length;
        if (k * 2 + 1 >= prizePositions[n - 1] - prizePositions[0]) {
            return n;
        }
        int ans = 0;
        int left = 0;
        int[] mx = new int[n + 1];
        for (int right = 0; right < n; right++) {
            while (prizePositions[right] - prizePositions[left] > k) {
                left++;
            }
            ans = Math.max(ans, right - left + 1 + mx[left]);
            mx[right + 1] = Math.max(mx[right], right - left + 1);
        }
        return ans;
    }

    /**
     * 2576. 求出最多标记下标
     * 给你一个下标从0开始的整数数组nums。
     * 一开始，所有下标都没有被标记。你可以执行以下操作任意次：
     * 选择两个互不相同且未标记的下标i和j,满足2*nums[i]<=nums[j],标记下标i和j。
     * 请你执行上述操作任意次，返回nums中最多可以标记的下标数目。
     * @param nums
     * @return
     */
    public static int maxNumOfMarkedIndices(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int ans = 0;
        int left = 0;
        for (int right = 0; right < n; right++) {
            while (left < n && nums[left] * 2 > nums[right]) {
                left++;
                ans += 2;
            }
        }
        return ans;
    }

    /**
     * 1833. 雪糕的最大数量
     * 夏日炎炎,小男孩 Tony 想买一些雪糕消消暑。商店中新到n支雪糕,
     * 用长度为n的数组costs表示雪糕的定价,其中costs[i]表示第i支雪糕的现金价格。Tony一共有
     * coins现金可以用于消费,他想要买尽可能多的雪糕。注意：Tony 可以按任意顺序购买雪糕。
     * 给你价格数组costs和现金量coins,请你计算并返回Tony用coins现金能够买到的雪糕的最大数量。
     * 你必须使用计数排序解决此问题。
     * @param costs
     * @param coins
     * @return
     */
    public static int maxIceCream(int[] costs, int coins) {
        Arrays.sort(costs);
        int sum = 0;
        int i =0;
        for (i =0 ;i <costs.length; i++) {
            sum += costs[i];
            if (sum == coins) {
                return i+1;
            } else if (sum > coins) {
                return i;
            }
        }
        return i;
    }

    /**
     * 3074. 重新分装苹果
     * 给你一个长度为n的数组apple和另一个长度为m的数组capacity。
     * 一共有n个包裹,其中第i个包裹中装着apple[i]个苹果。同时,还有m个箱子,第i个箱子的容量为
     * capacity[i]个苹果。请你选择一些箱子来将这n个包裹中的苹果重新分装到箱子中，
     * 返回你需要选择的箱子的最小数量。注意，同一个包裹中的苹果可以分装到不同的箱子中。
     * @param apple
     * @param capacity
     * @return
     */
    public static int minimumBoxes(int[] apple, int[] capacity) {
        int sum = 0;
        for (int i : apple) {
            sum += i;
        }
        Arrays.sort(capacity);
        int j = capacity.length-1;
        int num = 1;
        for (; j>=0; j--) {
            sum -= capacity[j];
            if (sum > 0 ){
                num++;
            } else if (sum <=0 ){
                break;
            }
        }
        return num;
    }

    /**
     * 2279. 装满石头的背包的最大数量
     * 现有编号从0到n-1的n个背包。给你两个下标从0开始的整数数组capacity和rocks。
     * 第i个背包最大可以装capacity[i]块石头，当前已经装了rocks[i]块石头。
     * 另给你一个整数additionalRocks,表示你可以放置的额外石头数量,石头可以往任意背包中放置。
     * 请你将额外的石头放入一些背包中，并返回放置后装满石头的背包的最大数量。
     * @param capacity
     * @param rocks
     * @param additionalRocks
     * @return
     */
    public static int maximumBags(int[] capacity, int[] rocks, int additionalRocks) {
        int n = capacity.length;
        List<Integer> list = new ArrayList<>();
        int ret = 0;
        for (int i = 0; i < n; i++) {
            if (capacity[i] <= rocks[i]) {
                ret++;
            } else {
                list.add(capacity[i] - rocks[i]);
            }
        }
        Collections.sort(list);
        for (Integer cnt : list) {
            if (additionalRocks >= cnt) {
                additionalRocks -= cnt;
                ret++;
            } else {
                break;
            }
        }
        return ret;
    }

    /**
     * 1481. 不同整数的最少数目
     * 给你一个整数数组arr和一个整数k。现需要从数组中恰好移除k个元素,
     * 请找出移除后数组中不同整数的最少数目。
     * @param arr
     * @param k
     * @return
     */
    public static int findLeastNumOfUniqueInts(int[] arr, int k) {
        Map<Integer, Integer> countMap = new HashMap<>();

        // 计算每个整数的出现频次
        for (int num : arr) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }

        // 将频次放入二维数组
        int[][] freqArray = new int[countMap.size()][2];
        int index = 0;
        for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
            freqArray[index][0] = entry.getKey();
            freqArray[index][1] = entry.getValue();
            index++;
        }

        // 按照频次升序排序
        Arrays.sort(freqArray, (a, b) -> Integer.compare(a[1], b[1]));

        // 移除频次最低的整数
        int uniqueCount = freqArray.length;
        for (int i = 0; i < freqArray.length; i++) {
            if (k >= freqArray[i][1]) {
                k -= freqArray[i][1];
                uniqueCount--;
            } else {
                break;
            }
        }

        return uniqueCount;
    }

    /**
     * 2390. 从字符串中移除星号
     * 给你一个包含若干星号*的字符串s。在一步操作中，你可以：选中s中的一个星号。
     * 移除星号左侧最近的那个非星号字符，并移除该星号自身。
     * 返回移除所有星号之后的字符串。
     * 注意：生成的输入保证总是可以执行题面中描述的操作。
     *     可以证明结果字符串是唯一的。
     * @param s
     * @return
     */
    public static String removeStars(String s) {
        StringBuilder sb = new StringBuilder(s);
        for (int i = 0; i < sb.length(); i++) {
            if ('*' == sb.charAt(i)) {
                for (int j = i-1; j >= 0; j--) {
                    if (sb.charAt(j) != '*') {
                        sb.deleteCharAt(j);
                        i--;
                        sb.deleteCharAt(i);
                        i--;
                        break;
                    }
                }
            }
        }
        return sb.toString();
    }

    /**
     * 1005. K 次取反后最大化的数组和
     * 给你一个整数数组nums和一个整数k，按以下方法修改该数组：
     * 选择某个下标i并将nums[i]替换为-nums[i]。
     * 重复这个过程恰好k次。可以多次选择同一个下标i。
     * 以这种方式修改数组后,返回数组可能的最大和。
     * @param nums
     * @param k
     * @return
     */
    public static int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int sum = 0;
        for (; k>0; ) {
            nums[0] = -nums[0];
            k--;
            Arrays.sort(nums);
        }
        for (int i = 0; i < n; i++) {
            sum += nums[i];
        }
        return sum;
    }

    /**
     * 1827. 最少操作使数组递增
     * 给你一个整数数组nums(下标从 0开始).每一次操作中,你可以选择数组中一个元素,并将它增加1.
     * 比方说,如果nums=[1,2,3],你可以选择增加nums[1]得到nums=[1,3,3]。
     * 请你返回使nums严格递增的最少操作次数。我们称数组nums是严格递增的，
     * 当它满足对于所有的0<=i<nums.length-1都有nums[i]<nums[i+1]。
     * 一个长度为1的数组是严格递增的一种特殊情况。
     * @param nums
     * @return
     */
    public static int minOperations(int[] nums) {
        int n = nums.length;
        int k = 0;
        for (int i = 1; i < n; i++) {
            while (nums[i-1] >= nums[i]) {
                nums[i]++;
                k++;
            }
        }
        return k;
    }

    /**
     * 1184. 公交站间的距离
     * 环形公交路线上有n个站，按次序从0到n-1进行编号。我们已知每一对相邻公交站之间的距离，
     * distance[i]表示编号为i的车站和编号为(i+1)%n的车站之间的距离。
     * 环线上的公交车都可以按顺时针和逆时针的方向行驶。
     * 返回乘客从出发点 start 到目的地 destination 之间的最短距离。
     * @param distance
     * @param start
     * @param destination
     * @return
     */
    public static int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start > destination) {
            int tmp = start;
            start = destination;
            destination = tmp;
        }
        int d1 = 0;
        int d2 = 0;
        for (int i = 0; i < distance.length; i++) {
            if (start <= i && i < destination) {
                d1 += distance[i];
            } else {
                d2 += distance[i];
            }
        }
        return Math.min(d1, d2);
    }

    /**
     * 2614. 对角线上的质数
     * 给你一个下标从0开始的二维整数数组nums。
     * 返回位于nums至少一条对角线上的最大质数。如果任一对角线上均不存在质数，
     * 返回0。注意：如果某个整数大于1,且不存在除1和自身之外的正整数因子，
     * 则认为该整数是一个质数。如果存在整数i,使得nums[i][i]=val或者
     * nums[i][nums.length-i-1]=val，则认为整数val位于nums的一条对角线上。
     * 在上图中，一条对角线是 [1,5,9] ，而另一条对角线是 [3,5,7] 。
     * @param nums
     * @return
     */
    public static int diagonalPrime(int[][] nums) {
            int n = nums.length, ans = 0;
            for (int i = 0; i < n; ++i) {
                int x = nums[i][i];
                if (x > ans && isPrime(x))
                    ans = x;
                x = nums[i][n - 1 - i];
                if (x > ans && isPrime(x))
                    ans = x;
            }
            return ans;
        }

        private static boolean isPrime(int n) {

            for (int i = 2; i * i <= n; ++i)

                if (n % i == 0)

                    return false;

            return n >= 2; // 1 不是质数

        }


    /**
     * 2332. 坐上公交的最晚时间
     * 给你一个下标从0开始长度为n的整数数组buses,其中buses[i]表示第i辆公交车的出发时间。
     * 同时给你一个下标从0开始长度为m的整数数组passengers，其中passengers[j]表示第j位乘客的到达时间。
     * 所有公交车出发的时间互不相同，所有乘客到达的时间也互不相同。
     * 给你一个整数capacity，表示每辆公交车最多能容纳的乘客数目。
     * 每位乘客都会搭乘下一辆有座位的公交车。如果你在y时刻到达，公交在x时刻出发，满足 y <= x  且公交没有满，那么你可以搭乘这一辆公交。最早 到达的乘客优先上车。
     * 返回你可以搭乘公交车的最晚到达公交站时间。你不能跟别的乘客同时刻到达。
     * 注意：数组buses和passengers不一定是有序的。
     * @param buses
     * @param passengers
     * @param capacity
     * @return
     */
    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        return 0;
    }

    /**
     * 2414. 最长的字母序连续子字符串的长度
     * 字母序连续字符串是由字母表中连续字母组成的字符串。换句话说，
     * 字符串"abcdefghijklmnopqrstuvwxyz"的任意子字符串都是字母序连续字符串。
     * 例如,"abc"是一个字母序连续字符串,而 "acb"和"za"不是。
     * 给你一个仅由小写英文字母组成的字符串s，返回其最长的字母序连续子字符串的长度。
     * @param s
     * @return
     */
    public static int longestContinuousSubstring(String s) {
        char[] str =  s.toCharArray();
        int ans = 1;
        int cnt = 1;
        for (int i = 1; i < s.length(); i++) {
            if (str[i-1] + 1 == str[i]) {
                ans = Math.max(ans, ++cnt);

            } else {
                cnt = 1;
            }
        }
        return ans;
    }

    /**
     * 204. 计数质数
     * 给定整数 n，返回所有小于非负整数n的质数的数量 。
     * @param n
     * @return
     */
    public static int countPrimes(int n) {
        int k = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime(i)) {
                k++;
            }
        }
        return k;
    }

    /**
     * 1014. 最佳观光组合
     * 给你一个正整数数组values，其中values[i]表示第i个观光景点的评分，
     * 并且两个景点i和j之间的距离为j-i。
     * 一对景点（i<j）组成的观光组合的得分为values[i]+values[j]+i-j ，
     * 也就是景点的评分之和减去 它们两者之间的距离。
     * 返回一对观光景点能取得的最高分。
     * @param values
     * @return
     * 枚举右，维护左
     */
    public static int maxScoreSightseeingPair(int[] values) {
        int ans = 0;
        int mx = values[0];
        for (int j = 1; j < values.length; j++) {
            ans = Math.max(ans, mx + values[j] - j);
            mx = Math.max(mx, values[j] + j);
        }
        return ans;
    }

    /**
     * 121. 买卖股票的最佳时机
     * 给定一个数组 prices，它的第i个元素prices[i]表示一支给定股票第i天的价格。
     * 你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。
     * 设计一个算法来计算你所能获取的最大利润。
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回0。
     * @param prices
     * @return
     */
    public static int maxProfit(int[] prices) {
        int ans = 0;
        int mx = prices[0];
        for (int i = 1; i < prices.length; i++) {
            ans = Math.max(prices[i] - mx, ans);
            mx = Math.min(prices[i],mx);
        }
        return ans;
    }

    /**
     * 122. 买卖股票的最佳时机 II
     * 给你一个整数数组prices，其中prices[i]表示某支股票第i天的价格。
     * 在每一天，你可以决定是否购买和或出售股票。你在任何时候最多只能持有
     * 一股股票。你也可以先购买，然后在同一天出售。
     * 返回 你能获得的 最大 利润 。
     * @param prices
     * @return
     */
    public static int maxProfit1(int[] prices) {
        int n = prices.length;
        int[][] memo = new int[n][2];
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }
        return dfs(memo, prices, n - 1, 0);
    }

    public static int dfs(int[][] memo, int[] prices, int i, int hold) {
        if (i < 0) {
            return hold == 1? Integer.MIN_VALUE:0;
        }
        if (memo[i][hold] != -1)  {
            return memo[i][hold];
        }
        if (hold == 1) {
            return memo[i][hold] = Math.max(dfs(memo, prices, i - 1, 1), dfs(memo, prices, i - 1, 0) - prices[i]);
        }
        return memo[i][hold] = Math.max(dfs(memo, prices,i-1,0), dfs(memo, prices,i-1,1)+prices[i]);
    }

    /**
     * 80. 删除有序数组中的重复项 II
     * 给你一个有序数组nums，请你原地删除重复出现的元素，
     * 使得出现次数超过两次的元素只出现两次，返回删除后数组的新长度。
     * 不要使用额外的数组空间，你必须在原地修改输入数组并在使用O(1)
     * 额外空间的条件下完成。
     * @param nums
     * @return
     */
    public static int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 2) {
            return n;
        }
        int slow = 2;
        int fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }

    /**
     * 283. 移动零
     * 给定一个数组nums，编写一个函数将所有0移动到数组的末尾，
     * 同时保持非零元素的相对顺序。
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     * @param nums
     */
    public static void moveZeroes(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (fast<nums.length) {
            if (nums[fast] != 0) {
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }
        for (int i = slow; i < nums.length; i++) {
            nums[i] = 0;
        }
        System.out.println(Arrays.toString(nums));
    }

    /**
     * 49. 字母异位词分组
     * 给你一个字符串数组，请你将字母异位词组合在一起。
     * 可以按任意顺序返回结果列表。
     * 字母异位词是由重新排列源单词的所有字母得到的一个新单词。
     * @param strs
     * @return
     */
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }

    /**
     * 128. 最长连续序列
     * 给定一个未排序的整数数组nums，找出数字连续的最长序列
     * （不要求序列元素在原数组中连续）的长度。
     * 请你设计并实现时间复杂度为O(n)的算法解决此问题。
     * @param nums
     * @return
     */
    public static int longestConsecutive(int[] nums) {
        int res = 0;
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        int seqLen;
        for (int num : numSet) {
            if (!numSet.contains(num-1)) {
                seqLen = 1;
                while (numSet.contains(++num)) seqLen++;
                res = Math.max(res, seqLen);
            }
        }
        return res;
    }

    /**
     * 2207. 字符串中最多数目的子序列
     * 给你一个下标从0开始的字符串text和另一个下标从0开始且长度为2的字符串pattern，
     * 两者都只包含小写英文字母。你可以在text中任意位置插入一个字符，这个插入的字符
     * 必须是pattern[0]或者pattern[1].注意,这个字符可以插入在text开头或者结尾的位置。
     * 请你返回插入一个字符后，text中最多包含多少个等于pattern的子序列。
     * 子序列指的是将一个字符串删除若干个字符后（也可以不删除）,剩余字符保持原本顺序得到的字符串。
     * @param text
     * @param pattern
     * @return
     */
    public static long maximumSubsequenceCount(String text, String pattern) {
        char x = pattern.charAt(0);
        char y = pattern.charAt(1);
        long l0 = 0;
        long l1 = 0;
        long k = 0;
        for (char c : text.toCharArray()) {
            if ( c== y) {
                k+=l0;
                l1++;
            }
            if (c == x) {
                l0++;
            }
        }
        return k + Math.max(l0, l1);
    }

    /**
     * 15. 三数之和
     * 给你一个整数数组nums，判断是否存在三元组[nums[i],nums[j],nums[k]]
     * 满足i!=j、i!=k且j!=k，同时还满足nums[i]+nums[j]+nums[k]==0。
     * 请你返回所有和为0且不重复的三元组。注意：答案中不可以包含重复的三元组。
     * @param nums
     * @return
     */
//    public static List<List<Integer>> threeSum(int[] nums) {
//        int n = nums.length;
//        Arrays.sort(nums);
//        Set<List<Integer>> set = new HashSet<>();
//        int m = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = i+1; j < n; j++) {
//                for (int k = j+1; k < n; k++) {
//                    List<Integer> myset = new ArrayList<>();
//                    if (nums[i] +nums[j] + nums[k] == 0) {
//                        myset.add(nums[i]);
//                        myset.add(nums[j]);
//                        myset.add(nums[k]);
//                    }
//                    set.add(myset);
//                }
//            }
//        }
//        List<List<Integer>> list = new ArrayList<>();
//        for (List<Integer> integers : set) {
//            if (integers.size() == 0) {
//                continue;
//            }
//            list.add(integers);
//        }
//        return list;
//    }
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length-2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int left = i+1, right = nums.length-1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }

    /**
     * 160. 相交链表
     * 给你两个单链表的头节点headA和headB，请你找出并返回两个单链表相交的起始节点。
     * 如果两个链表不存在相交节点，返回 null。
     * 图示两个链表在节点c1开始相交：
     * 题目数据保证整个链式结构中不存在环。
     * 注意，函数返回结果后，链表必须 保持其原始结构 。
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode l1 = headA;
        ListNode l2 = headB;
        while (l1 != l2) {
            if (l1 == null) l1 = headB;
            else l1 = l1.next;
            if (l2 == null) l2 = headA;
            else l2 = l2.next;
        }
        return l1;
    }

    /**
     * 94. 二叉树的中序遍历
     * 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        inorder(root, list);
        return list;
    }

    public void inorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }

    /**
     * 2306. 公司命名
     * 给你一个字符串数组ideas表示在公司命名过程中使用的名字列表。公司命名流程如下：
     * 从ideas中选择2个不同名字，称为ideaA和ideaB。交换ideaA和ideaB的首字母。
     * 如果得到的两个新名字都不在ideas中,那么ideaA ideaB（串联ideaA和ideaB，
     * 中间用一个空格分隔）是一个有效的公司名字。否则，不是一个有效的名字。
     * 返回不同且有效的公司名字的数目。
     * @param ideas
     * @return
     */
    public static long distinctNames(String[] ideas) {
        Set<String>[] groups  = new HashSet[26];
        Arrays.setAll(groups, i->new HashSet<>());
        for (String idea : ideas) {
            groups[idea.charAt(0) - 'a'].add(idea.substring(1));
        }
        long ans = 0;
        for (int a = 1; a < 26; a++) {
            for (int b = 0; b < a; b++) {
                int m = 0;
                for (String s : groups[a]) {
                    if (groups[b].contains(s)) {
                        m++;
                    }
                }
                ans += (long) (groups[a].size() - m) * (groups[b].size() - m);
            }
        }
        return ans * 2;
    }

    /**
     * 213. 打家劫舍 II
     * 你是一个专业的小偷,计划偷窃沿街的房屋,每间房内都藏有一定的现金.
     * 这个地方所有的房屋都围成一圈,这意味着第一个房屋和最后一个房屋是紧挨着的。
     * 同时,相邻的房屋装有相互连通的防盗系统,如果两间相邻的房屋在同一晚上被小偷闯入,
     * 系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组,
     * 计算你在不触动警报装置的情况下,今晚能够偷窃到的最高金额
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        return 0;
    }

    /**
     * 23. 合并K个升序链表
     * 给你一个链表数组，每个链表都已经按升序排列。
     * 请你将所有链表合并到一个升序链表中，返回合并后的链表。
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) {
            return null;
        }
        //1. 处理头节点的特殊情况
        //2. 简化链表操作
        //3. 保持指针的一致性
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        PriorityQueue<ListNode> queue = new PriorityQueue<ListNode>(
                lists.length, (a,b) -> (a.val - b.val));
        for (ListNode list : lists) {
            if (list != null) {
                queue.add(list);
            }
        }
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            p.next = node;
            if (node.next != null) {
                queue.add(node.next);
            }
            p = p.next;
        }
        return dummy.next;
    }

    /**
     * 92. 反转链表 II
     * 给你单链表的头指针head和两个整数left和right,其中left<=right。
     * 请你反转从位置left到位置right的链表节点,返回反转后的链表。
     * @param head
     * @param left
     * @param right
     * @return
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == 1) {
            return reverseN(head, right);
        }
        head.next = reverseBetween(head.next, left-1, right-1);
        return head;
    }

    static ListNode node = null;
    public static ListNode reverseN(ListNode head, int n) {
        if (n==1) {
            node = head.next;
            return head;
        }
        ListNode last = reverseN(head.next, n-1);
        head.next.next = head;
        head.next = node;
        return last;
    }

    /**
     * 2073. 买票需要的时间
     * 有n个人前来排队买票，其中第0人站在队伍最前方,第(n-1)人站在队伍最后方
     * 给你一个下标从0开始的整数数组tickets,数组长度为n,其中第i人想要购买的票数为 tickets[i] 。
     * 每个人买票都需要用掉恰好1秒.一个人一次只能买一张票,如果需要购买更多票，
     * 他必须走到队尾重新排队(瞬间发生,不计时间).如果一个人没有剩下需要买的票，那他将会离开队伍。
     * 返回位于位置 k（下标从 0 开始）的人完成买票需要的时间（以秒为单位）
     * @param tickets
     * @param k
     * @return
     */
    public static int timeRequiredToBuy(int[] tickets, int k) {
        int ans = 0;
        for (int i = 0; i < tickets.length; i++) {
            if (i <= k) {
                ans += Math.min(tickets[i], tickets[k]);
            } else {
                ans += Math.min(tickets[i], tickets[k]-1);
            }
        }
        return ans;
    }

    /**
     * 3. 无重复字符的最长子串
     * 给定一个字符串s，请你找出其中不含有重复字符的最长子串的长度。
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<Character>();
        int n = s.length();
        int rk = -1, ans = 0;
        for (int i = 0; i < n; i++) {
            if (i != 0) {
                set.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !set.contains(s.charAt(rk + 1))) {
                set.add(s.charAt(rk + 1));
                ++rk;
            }
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }

    /**
     * 983. 最低票价
     * 在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，
     * 你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。
     * 火车票有 三种不同的销售方式 ：
     *     一张 为期一天 的通行证售价为 costs[0] 美元；
     *     一张 为期七天 的通行证售价为 costs[1] 美元；
     *     一张 为期三十天 的通行证售价为 costs[2] 美元。
     * 通行证允许数天无限制的旅行。例如，如果我们在第2天获得一张为期7天的通行证，
     * 那么我们可以连着旅行7天：第2天、第3天、第4天、第5天、第6天、第7天和第8天。
     * 返回 你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费 。
     * @param days
     * @param costs
     * @return
     */
    public int mincostTickets(int[] days, int[] costs) {
        int lastDay = days[days.length-1];
        boolean[] isTravel = new boolean[lastDay+1];
        for (int day : days) {
            isTravel[day] = true;
        }
        int[] memo = new int[lastDay+1];
        return dfs(isTravel, costs, lastDay, memo);
    }

    public int dfs(boolean[] isTravel, int[] costs, int i, int[] memo) {
        if(i<=0) {
            return 0;
        }
        if(memo[i] > 0) {
            return memo[i];
        }
        if(!isTravel[i]) {
            return memo[i] = dfs(isTravel,costs,i-1,memo);
        }
        return memo[i] = Math.min(dfs(isTravel,costs,i-1,memo)+costs[0],
                Math.min(dfs(isTravel,costs,i-7,memo)+costs[1],
                        dfs(isTravel,costs,i-30,memo)+costs[2]));
    }

    /**
     * 1870. 准时到达的列车最小时速
     * 给你一个浮点数hour,表示你到达办公室可用的总通勤时间.要到达办公室,
     * 你必须按给定次序乘坐n趟列车.另给你一个长度为n的整数数组dist,其中dist[i]
     * 表示第i趟列车的行驶距离（单位是千米）。每趟列车均只能在整点发车,
     * 所以你可能需要在两趟列车之间等待一段时间。例如，第1趟列车需要1.5小时,
     * 那你必须再等待0.5小时,搭乘在第2小时发车的第2趟列车。
     * 返回能满足你准时到达办公室所要求全部列车的最小正整数时速（单位：千米每小时），
     * 如果无法准时到达,则返回-1.生成的测试用例保证答案不超过107,且hour的小数点后最多存在两位数字 。
     * @param dist
     * @param hour
     * @return
     */
    public int minSpeedOnTime(int[] dist, double hour) {
        int n = dist.length;
        long h100 = Math.round(hour*100);
        long delta = h100 - (n-1) * 100;
        if (delta <= 0) {
            return -1;
        }
        int maxDist = 0;
        for (int d : dist) {
            maxDist = Math.max(maxDist,d);
        }
        if (h100 <= n*100) {
            return Math.max(maxDist, (int) ((dist[n-1] * 100 - 1)/delta+1));
        }
        int left = 0;
        int h = (int) (h100/(n*100));
        int right = (maxDist -1) /h +1;
        while (left+1 < right) {
            int mid = (left+right) >>> 1;
            if (check(mid, dist, h100)) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return right;
    }

    private boolean check(int v, int[] dist, long h100) {
        int n = dist.length;
        long t = 0;
        for (int i = 0; i < n - 1; i++) {
            t += (dist[i]-1)/v+1;
        }
        return (t*v+dist[n-1]) * 100 <= h100 * v;
    }

    /**
     * 1227. 飞机座位分配概率
     * 有n位乘客即将登机,飞机正好有n个座位.第一位乘客的票丢了,他随便选了一个座位坐下。
     * 剩下的乘客将会：如果他们自己的座位还空着，就坐到自己的座位上，
     * 当他们自己的座位被占用时，随机选择其他座位.第n位乘客坐在自己的座位上的概率是多少？
     * @param n
     * @return
     */
    public double nthPersonGetsNthSeat(int n) {
        return n ==1 ? 1 : 0.5;
    }

    /**
     * 1436. 旅行终点站
     * 给你一份旅游线路图，该线路图中的旅行线路用数组paths表示，其中paths[i]=[cityAi,cityBi]
     * 表示该线路将会从cityAi直接前往cityBi。请你找出这次旅行的终点站，
     * 即没有任何可以通往其他城市的线路的城市。
     * 题目数据保证线路图会形成一条不存在循环的线路，因此恰有一个旅行终点站。
     * @param paths
     * @return
     */
    public static String destCity(List<List<String>> paths) {
        Set setA = new HashSet();
//        Set setB = new HashSet();
//        String a  = new String();
        for (List<String> path : paths) {
            setA.add(path.get(0));
//            setB.add(path.get(1));
        }
//        for (Object o : setB) {
//            if (!setA.contains(o)) {
//                a = o.toString();
//            }
//        }
        for (List<String> path : paths) {
            if (!setA.contains(path.get(1))){
                return path.get(1);
            }
        }
        return paths.get(0).get(1);
    }

    /**
     * 3171. 找到按位或最接近K的子数组
     * 给你一个数组nums和一个整数k.你需要找到nums的一个子数组,满足子数组中所有元素按位或运算OR
     * 的值与k的绝对差尽可能小.换言之,你需要选择一个子数组nums[l..r]满足
     * |k-(nums[l] OR nums[l + 1]...OR nums[r])|最小。
     * 请你返回最小的绝对差值。
     * 子数组是数组中连续的非空元素序列。
     * @param nums
     * @param k
     * @return
     */
    public static int minimumDifference(int[] nums, int k) {
        int ans = Integer.MAX_VALUE;
        int left = 0;
        int bottom = 0;
        int rightOr = 0;
        for (int right = 0; right < nums.length; right++) {
            rightOr |= nums[right];
            while (left <= right && (nums[left] | rightOr) > k) {
                ans = Math.min(ans, (nums[left] | rightOr) - k);
                if (bottom <= left) {
                    for (int i = right - 1; i > left; i--) {
                        nums[i] |= nums[i + 1];
                    }
                    bottom = right;
                    rightOr = 0;
                }
                left++;
            }
            if (left <= right) {
                ans = Math.min(ans, k - (nums[left] | rightOr));
            }
        }
        return ans;
    }

    /**
     * 67. 二进制求和
     * 给你两个二进制字符串a和b，以二进制字符串的形式返回它们的和。
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        return Integer.toBinaryString(
                Integer.parseInt(a, 2) + Integer.parseInt(b , 2)
        );
    }


    public static void main(String[] args) {
        int[] head = {3, 2, 0, -4};
        int pos = 1;
        int[] rocks = {14,13,16,44,8,20,51,15,46,76,51,20,77,13,14,35,6,34,34,13,3,8,1,1,61,5,2,15,18};
        int[] capacity = {54,18,91,49,51,45,58,54,47,91,90,20,85,20,90,49,10,84,59,29,40,9,100,1,64,71,30,46,91};
        int additionalRocks = 77;
        String s = "abcde";
        int coins = 20;
        String[] ideas  = {"coffee","donuts","time","toffee"};
        int start = 0, destination = 2;
        ListNode node = convertArrayToList(head);
        int[] tickets = {5,1,1,1};
        String text = "aabb", pattern = "ab";
        String[][] pathsArray = {
                {"B","C"},
                {"D","B"},
                {"C","A"}
        };

        int[] nums1 = {1,2,4,12}, nums2 = {2,4};
        int k = 3;
        //System.out.println(numberOfPairs(nums1, nums2, k));
        // 转换为 List<List<String>>
        List<List<String>> paths = new ArrayList<>();
        for (String[] path : pathsArray) {
            paths.add(Arrays.asList(path));
        }
        int[] nums = {1,2,2,1};
        //System.out.println(duplicateNumbersXOR(nums));
        //printList(removeNthFromEnd(node,1));
//        System.out.println(destCity(paths));
    }
}
