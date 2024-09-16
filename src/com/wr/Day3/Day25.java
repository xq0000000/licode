package com.wr.Day3;

public class Day25 {
    public static int minimumSwap(String s1, String s2) {
        int xy = 0, yx = 0;
        int n = s1.length();
        for (int i = 0; i < n; i++) {
            char a = s1.charAt(i), b = s2.charAt(i);
            if (a == 'x' && b == 'y') {
                xy++;
            }
            if (a == 'y' && b == 'x') {
                yx++;
            }
        }
        if ((xy + yx) % 2 == 1) {
            return -1;
        }
        return xy / 2 + yx / 2 + xy % 2 + yx % 2;
    }

    public static void main(String[] args) {
        String s1 = "xxyxy";
        String s2 = "xyxyx";
        System.out.println(minimumSwap(s1,s2));
    }
}
