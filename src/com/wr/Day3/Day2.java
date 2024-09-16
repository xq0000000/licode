package com.wr.Day3;

public class Day2 {
    public static boolean validateStackSequences(int[] pushed, int[] popped) {
        int[] stack = new int[pushed.length];
        int k = 0;
        for (int i = 0,j=0; i < pushed.length; i++) {
            stack[k++] = pushed[i];
            while(k>0&&stack[k - 1] == popped[j]){
                    k--;
                    j++;
            }
        }
        return k==0;
    }

    public static void main(String[] args) {
        int[] pushed=new int[]{1,2,3,4,5};
        int[] popped=new int[]{4,5,3,2,1};
        System.out.println(validateStackSequences(pushed,popped));
    }
}
