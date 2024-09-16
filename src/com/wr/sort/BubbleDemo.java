package com.wr.sort;

import java.util.Arrays;

/**
 * ClassName: BubbleDemo
 * Description: 冒泡排序
 * date: 2024/8/8 10:17
 * 重复地走访过要排序的数列，一次比较两个元素，如果它们的顺序错误就把它们交换过来
 * @author Wang
 * @since JDK 1.8
 */
public class BubbleDemo {
    public static void main(String[] args) {
        int[] a = {4,7,2,1,4,6,8};
        bubbleSort(a);
        System.out.println(Arrays.toString(a));
    }

    public static void bubbleSort(int[] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = i; j < a.length; j++) {
                if (a[i] > a[j]) { // 一个一个对比交换
                    int temp = a[i];
                    a[i] = a[j];
                    a[j] = temp;
                }
            }
        }
    }
}
