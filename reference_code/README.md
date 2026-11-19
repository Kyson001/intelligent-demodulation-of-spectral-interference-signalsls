# 原始参考代码与本地修改快照

`local_plotting_original/`保存本次上传前用户本地四个绘图脚本的字节一致快照。其中仍有原来的桌面绝对路径。可直接在仓库运行的版本位于 `scripts/`，只适配了数据、预处理模块和输出路径，保留绘图内容及训练数据一致性检查。

`teacher_original/`保存老师提供的三个原始`.m`文件。它们作为参考材料原样归档，没有纳入已验证的Python数据与训练流程。`spectrum_generation_test.m`含Python与MATLAB混合语法，不能当作已验证可运行的MATLAB脚本；`quadratic_fit_for_SPI.m`还使用`gaussmf`，需对应MATLAB工具箱。
