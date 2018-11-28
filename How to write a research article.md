# How to write a research article
## Tools
- Latex
  - Windows: TexStudio + Miktex
  - Mac: sublime text 3
- grammerly.com
  帮忙改掉非常小的语法错误
- google schalor
- cibo.cn

(Before paper writing, scan through the guideline carefully. eg: CVPR guideline)

## Paper Structure
- Abstract
- Introduction
- Related Works
- Approach
- Experimental Results
- Conclusion

## Abstract
- 8-11 sentences
  - Problem (1)
  - Challenges (2-3)
  - How we solve (3-4)
  - Summarize experimental results (2-3)
- 浓缩版的introduction

## Introduction
- What, How, Why
  - Describing background of the application in this paper
  - Analyzing the difficulties
  - How existing works solved it?
  - What is the problem of existing works?
  - How we plan to solve this problem?
  - Introduce our method (coarse to fine)
  - What experiments we have conducted?
  - Summarizing our contributions
- 整篇文章最精华的部分
- 你受什么启发
- 注意别人能不能看得懂，假设读者是完全没有接触过这个领域的人，一步一步引领读者
- why：在学术界，唯结果论是不行的，一定要convince
- 一个introduction写得好不好，是影响reviewer最关键的东西（一般来说建议最后再写）

## Related Works
- 一般来说包含两个部分
  - Related works of the application in this paper
  - Related works of methods which used in this paper
- Bibtex in *.bib
- \cite{}
- citation不能直接作为一个名词,不能放在句首
```Latex
\cite{he2016cvpr} proposed …  ->  [1] proposed … (x)
The authors of~\cite{he2016cvpr} … ->  The authors of [1]…  (√)
He~\etal~\cite{he2016cvpr} … ->  He et al. [1] …
He and Han~\cite{he2016cvpr} … -> He and Han [1] … (如果主作者有两个人co-auther)
```
- 利用google schalor来写related works, 搜一篇比较好的引用几百的，然后related works从它的这些引用中找。
- cvpr的citation一定要一页塞满

## Approach
- how and why
  - System overview
  - Logic and connections 
  - Intuition
  - Guide the audience
- 最好写section进来（参照section3.1这类的论述），section与section之间一定要衔接好。
- 重要的是intuition，也就是why，让读者信服
- 要老板来改之前一定要清晰通顺

## Experimental Results
- Introduce the setting in the experiments
- Step-by-step, easy-to-hard
- Connections among the experiments
- Analyzing the reason
**注意**
- 怎么样选对要做什么样的实验
- 实验component的排列组合
- 循序渐进：最开始跟几个baseline比较
- 最后很重要的放一些visualization的结果
- 有些figure没那么重要，把它放在单栏
- 有些figure比较重要，把它放在双栏
- siggraph比较重视图片，cv比较重视统计
- 一定要有定性的，定量的

## Conclusion
- Contributions of the paper
- Future work
- More discussions

**一篇paper据统计大约300-400句**

## Title
- 不要是一个句子

## Figure
- Teaser
  - tesser是整个一页的第一张图
  - 作用：show off 我们的technique有多强，选最好的结果就行了
  - 通常tesser的图片，caption会解释的特别多
- Functional figure
  - 解释文字上说不清楚的图
- System overview
- Results
**一些细节**
```Latex
\includegraphics[width=\linewidth]{***.png/pdf} No JPEG!
width=\linewidth 自动缩放成column一样宽
```
- Proper content
- Good layout
- Caption: What is this figure? & Describing the figure (if needed)
- Reference:
```Latex
  Figure~\ref{figure-label} or Fig~\ref{figure-label}. Just be consistent. 
  Figure~\ref{figure-label}(a)~&~(b)
```
## Table
- Online latex figure generator 自动生成图表
- \hline
- \vline
- \multicolumn{number of columns}{format, e.g., c/l/r }{content}

## Equation
- Clearer notation, notation一定要清晰
- Do not use the same notation for different purposes
  - One letter for each variable
- \rm : \mathcal{L}_{\rm Siamese} = …

## Consistency
- Terms
  - Pyramidal ……network, Multiscale ……network 
- Capital
  - Pyramidal network, pyramidal network
- Notation

**一些用词的term一定要一致，大小写一定要一致，细节的地方一定要注意**

**写完一篇paper，给老师改完之后，一定要对比老师改之前的version与老师改之后的version，这样才能有所提升。**
