项目代码是我给你发的Github项目，ARNG_Generator，这个项目的idea出自于上传的论文，Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic

目前这个方案生成的图就是上传的negative_example这张图，你可以看到从根节点到可推理的叶子节点，路径(depth)通常都很短，这么短的推理路径意味着构建的数据集逻辑强度也不高

而我想要的其实是positive_example这张图你可以看到从var1到最终的And(And(And(Var0, Var1), Var2), Var3),推理路径是3，虽然说路径也不是很长，但结构更趋向于一个结构较为复杂的图

我想要的最终生成的图结构和推理结构，要求要很复杂，通常情况下depth可能在5~10，甚至要超过10

具体你可以查看我上传的论文,learning deductive reasoning from synthetic corpus based on formal logic的figure 3还有table 1中FLD的内容，FLD的proof tree depth在我这篇论文中对应的就应该是depth，而proof tree branches就应该是我这篇论文的max_branching，这两个应该是一个东西，如果不是一样的，那就是现在的方案有问题

请你根据我提供的文字内容、图还有论文，逐个分析现在项目存在的问题；并给出对应的解决方案

