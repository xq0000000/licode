package com.wr.Mouth9;

import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * ClassName: Week1
 * Description:
 * date: 2024/9/22 10:34
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Week1 {
    /**
     * Q1. 举报垃圾信息
     * 给你一个字符串数组message和一个字符串数组bannedWords。
     * 如果数组中至少存在两个单词与bannedWords中的任一单词完全相同，
     * 则该数组被视为垃圾信息。如果数组message是垃圾信息,则返回 true；
     * 否则返回 false。
     * @param message
     * @param bannedWords
     * @return
     */
    public static boolean reportSpam(String[] message, String[] bannedWords) {
        int k = 0;
        Set<String> stringSet = Arrays.stream(bannedWords)
                .collect(Collectors.toSet());
        for (int i = 0; i < message.length; i++) {
                if (stringSet.contains(message[i])) {
                    k++;
            }
        }
        return k>=2?true:false;
    }

    /**
     * Q2. 移山所需的最少秒数
     * 给你一个整数 mountainHeight表示山的高度。
     * 同时给你一个整数数组workerTimes，表示工人们的工作时间（单位：秒）。
     * 工人们需要同时进行工作以降低山的高度。对于工人i:
     * 山的高度降低 x，需要花费workerTimes[i]+workerTimes[i]*2+...+workerTimes[i]*x秒。例如：
     * 山的高度降低 1，需要workerTimes[i]秒。
     * 山的高度降低 2，需要workerTimes[i]+workerTimes[i]*2秒，依此类推。
     * 返回一个整数，表示工人们使山的高度降低到 0 所需的 最少 秒数。
     * @param mountainHeight
     * @param workerTimes
     * @return
     */
    // mountainHeight = 4, workerTimes = [2,1,1]
    // x + y + z = mountainHeight; Math.min(dp[i], dp[i-1])
    // dp[i]:移动i层山需要的时间
    public long minNumberOfSeconds(int mountainHeight, int[] workerTimes) {
        long[] workerTimesInLong = new long[workerTimes.length];
        long[] mul = new long[workerTimes.length];
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> (Long.compare(workerTimesInLong[a], workerTimesInLong[b])));
        for (int i = 0; i < workerTimesInLong.length; i++) {
            workerTimesInLong[i] = workerTimes[i];
            mul[i] = 1;
            queue.add(i);
        }
        long res = 0;
        while (mountainHeight > 0 && !queue.isEmpty()) {
            int tmp = queue.poll();
            res = Math.max(res, workerTimesInLong[tmp]);
            mul[tmp] += 1;
            workerTimesInLong[tmp] = workerTimesInLong[tmp] + mul[tmp] * workerTimes[tmp];
            queue.add(tmp);
            --mountainHeight;
        }
        return res;
    }

    public static void main(String[] args) {
        String[] message = {"hello","programming","fun"};
        String[] bannedWords = {"world","programming","leetcode"};
        System.out.println(reportSpam(message, bannedWords));

    }
}
