package com.wr.Mouth9;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * ClassName: Week2
 * Description:
 * date: 2024/9/29 10:39
 *
 * @author Wang
 * @since JDK 1.8
 */
public class Week2 {

    /**
     * Q1. 找出第 K 个字符 I
     * Alice和Bob正在玩一个游戏。最初，Alice有一个字符串 word = "a"。
     * 给定一个正整数 k。
     * 现在 Bob会要求 Alice执行以下操作 无限次 :
     * 将word中的每个字符更改为英文字母表中的下一个字符来生成一个新字符串，并将其追加到原始的word。
     * 例如，对 "c" 进行操作生成 "cd"，对 "zb" 进行操作生成 "zbac"。
     * 在执行足够多的操作后， word 中 至少 存在 k 个字符，此时返回 word 中第 k 个字符的值。
     * 注意，在操作中字符 'z' 可以变成 'a'。
     * @param k
     * @return
     */
    public static char kthCharacter(int k) {
        String word = "a";
        StringBuilder word1 = new StringBuilder();
        word1.append(word);
        while (word1.length()<k) {
            String flag = "";
            for (int i = 0; i < word.length(); i++) {
                word1.append((char) (word.charAt(i) + 1));
                flag+=(char) (word.charAt(i) + 1);

            }
            word+=flag;
        }
        return word1.charAt(k-1);
    }

    /**
     * Q2. 元音辅音字符串计数 I
     * 给你一个字符串 word 和一个 非负 整数 k。
     * 返回 word的子字符串中，每个元音字母（'a'、'e'、'i'、'o'、'u'）至少
     * 出现一次，并且恰好包含k个辅音字母的子字符串的总数。
     * @param word
     * @param k
     * @return
     */
    public static int countOfSubstrings(String word, int k) {
        Set set = new HashSet();
        set.add('a');
        set.add('e');
        set.add('i');
        set.add('o');
        set.add('u');
        int n = word.length();
        int count = 0;

        // 遍历所有可能的子字符串
        for (int start = 0; start < n; start++) {
            int vowelCount = 0;
            int consonantCount = 0;
            Set<Character> foundVowels = new HashSet<>();

            for (int end = start; end < n; end++) {
                char ch = word.charAt(end);

                if (set.contains(ch)) {
                    foundVowels.add(ch);
                    vowelCount++;
                } else {
                    consonantCount++;
                }

                if (foundVowels.size() == 5 && consonantCount == k) {
                    count++;
                }
            }
        }

        return count;
    }

    public static void main(String[] args) {
        System.out.println(countOfSubstrings("ieaouqqieaouqq",1));
    }
}
