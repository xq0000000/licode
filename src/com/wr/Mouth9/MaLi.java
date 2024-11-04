package com.wr.Mouth9;

import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

/**
 * ClassName: MaLi
 * Description:
 * date: 2024/9/24 19:18
 *
 * @author Wang
 * @since JDK 1.8
 */
public class MaLi {
    public static void main(String[] args) {

    }

    public static void main1(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String str = scanner.nextLine();
        int flag = 0;
        int[] count = new int[256];
        Set set = new HashSet();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            count[c]++;
            if (count[c] > 1 && !set.contains(str.charAt(i))) {
                for (int j = 0; j < str.length(); j++) {
                    if (str.charAt(j) == c) {
                        int k = j + 1;
                        System.out.print(str.charAt(j) + ", " + k + "; ");
                        set.add(str.charAt(i));
                    }
                }
            }
        }
    }
}
