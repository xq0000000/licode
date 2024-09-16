package com.wr.sort;

import java.util.Arrays;

/**
 * ClassName: InsertionDemo
 * Description: 插入排序
 * date: 2024/8/8 14:22
 * 将数组分为已排序部分和未排序部分，逐步将未排序部分的
 * 元素插入到已排序部分的合适位置，直到整个数组有序为止
 * @author Wang
 * @since JDK 1.8
 */
public class InsertionDemo {

    public static void main(String[] args) {
        int[] a = {4,7,2,1,4,6,8};
        insertionSort(a);
        System.out.println(Arrays.toString(a));
    }

    public static void insertionSort(int[] a) {
        for (int i = 0; i < a.length; i++) {
            int temp = a[i]; //将当前待排序元素存储在临时变量temp中
            int index = i; //记录当前待排序元素的位置
            while (index > 0 && a[index-1] > temp) { //如果已经排序好的元素比待排序的大
                a[index] = a[index-1]; //将已经排序好的内容往后一位，放到待排序的位置上
                index--; //再依次和已经排序好的内容进行比较
            }
            a[index] = temp; //最后把待排序的内容放到正确的位置上
        }
    }
}
