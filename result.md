\section{Legal Clause Performance:} The Retrieval-Grounded Reasoning system achieves the highest accuracy across all legal clause metrics. The most notable improvement is observed in the strict \texttt{full\_signature\_law\_f1}, where it achieves 0.4893 compared to 0.2821 for Single-Step and 0.2954 for Past-Case Reasoning. This indicates that explicitly retrieving and reasoning over multi-step law text is crucial for identifying precise clause and point-level legal bases, not just broad articles.
\begin{table}[htbp]
\centering
\small
\caption{Legal Clause Metrics. Higher is better.}
\label{tab:eval_legal}
\begin{tabular}{|l|r|r|r|}
\hline
\textbf{Metric} &
\makecell[c]{\textbf{Retrieval-}\\\textbf{Grounded}\\\textbf{Reasoning}} &
\makecell[c]{\textbf{Single-Step}\\\textbf{Reasoning}} &
\makecell[c]{\textbf{Past-Case}\\\textbf{Reasoning}} \\
\hline
Article-level law \(F_1\) & 0.6197 & 0.5896 & 0.3861 \\
Full-signature law \(F_1\) & 0.4893 & 0.2821 & 0.2954 \\
Offence-article \(F_1\) & 0.6561 & 0.6201 & 0.4150 \\
Exact article-set match rate & 0.1020 & 0.0889 & 0.0889 \\
\hline
\end{tabular}
\end{table}

\section{Sentence Prediction Performance} The multi-step reasoning pipeline significantly outperforms the baselines in predicting sentence length. The Retrieval-Grounded method minimizes both MAE (34.48) and RMSE (35.48). Single-Step Reasoning sits in the middle with a MAE of 57.00, while Past-Case Reasoning struggles heavily with the highest MAE of 73.63, suggesting that carelessly injecting past cases without structured evaluation may confuse the model regarding sentencing duration, despite having a slightly lower RMSE than the single-step baseline.

\begin{table}[htbp]
\centering
\small
\caption{Sentence Metrics. Lower is better.}
\label{tab:eval_sentence}
\begin{tabular}{|l|r|r|r|}
\hline
\textbf{Metric} &
\makecell[c]{\textbf{Retrieval-}\\\textbf{Grounded}\\\textbf{Reasoning}} &
\makecell[c]{\textbf{Single-Step}\\\textbf{Reasoning}} &
\makecell[c]{\textbf{Past-Case}\\\textbf{Reasoning}} \\
\hline
Sentence MAE in months  & 34.48 & 57.00 & 73.63 \\
Sentence RMSE in months & 35.48 & 59.91 & 53.97 \\
\hline
\end{tabular}
\end{table}

\section{Conclusion Alignment Performance} This group yields fascinating insights. While the Retrieval-Grounded and Single-Step systems hover around 68\% accuracy for Offense Alignment (\texttt{Toi\_Danh}), the Past-Case Reasoning method excels here, reaching 84.71\%. This suggests that directly retrieving past judgments is highly effective for identifying the semantic category or textual description of the crime. However, for Penalty/Fine Alignment (\texttt{Phat\_Tien}), all three methods perform similarly well, ranging tightly between 85\% and 88\%.