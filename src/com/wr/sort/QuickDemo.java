package com.wr.sort;

import java.util.Arrays;

/**
 * ClassName: QuickDemo
 * Description: 快速排序
 * date: 2024/8/8 11:10
 * 根据base将数组分为两块，小于base都放到左边，大于base都放到右边。然后递归排序
 * @author Wang
 * @since JDK 1.8
 */
public class QuickDemo {
    public static void main(String[] args) {
        int[] a = {4,7,2,1,4,6,8};
        QuickSort(a,  0, a.length - 1);
        System.out.println(Arrays.toString(a));
    }

    public static void QuickSort(int[] a, int left, int right) {
        if (left > right) {
            return;
        }
        int base = a[left];
        int l = left;
        int r = right;
        while (l!=r) {
            while (a[r] >= base && r>l) {
                r--;
            }
            while (a[l] <= base && r>l) {
                l++;
            }
            int temp = a[l];
            a[l] = a[r];
            a[r] = temp;
        }
        a[left] = a[l];
        a[l] = base;
        QuickSort(a, left, l-1);
        QuickSort(a, l+1, right);
    }
}
