package com.wr.Mouth9;

import java.util.PriorityQueue;

/**
 * ClassName: cla
 * Description:
 * date: 2024/9/30 9:14
 * 1845. 座位预约管理系统
 * 请你设计一个管理n个座位预约的系统，座位编号从1到n。
 * 请你实现SeatManager类：
 * SeatManager(int n)初始化一个SeatManager对象,它管理从1到n编号的n个座位。
 * 所有座位初始都是可预约的。int reserve()返回可以预约座位的最小编号,此座位变为不可预约。
 * void unreserve(int seatNumber)将给定编号seatNumber对应的座位变成可以预约。
 * @author Wang
 * @since JDK 1.8
 */
public class SeatManager {

    private final PriorityQueue<Integer> available =  new PriorityQueue<>();

    public SeatManager(int n) {
        for (int i = 1; i <= n; i++) {
            available.add(i);
        }
    }

    public int reserve() {
        return available.poll();
    }

    public void unreserve(int seatNumber) {
        available.add(seatNumber);
    }


}
