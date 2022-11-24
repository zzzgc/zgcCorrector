from hashlib import shake_128
import re
import operator

def stastics(sents1, sents2, sents3):
    """
    该纠的，即有错文本记为 P，不该纠的，即无错文本记为 N
    对于该纠的，纠对了，记为 TP，纠错了或未纠，记为 FP
    对于不该纠的，未纠，记为 TN，纠了，记为 FN。
    :param sents1: input
    :param sents2: output
    :param sents3: target
    :return: F_sent
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    TP_sent, TN_sent, FP_sent, FN_sent = 0, 0, 0, 0

    for sent1, sent2, sent3 in zip(sents1, sents2, sents3):
        # sent1 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
        #                "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
        #                "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
        #                "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent1)
        # sent1 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent1)
        # sent2 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
        #                "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
        #                "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
        #                "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent2)
        # sent2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent2)
        # sent3 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
        #                "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
        #                "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
        #                "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent3)
        # sent3 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent3)
        # if sent1 == sent2 and sent1 == sent3:
        #     continue
        for i in range(len(sent1)):
            if sent1[i] == 0:
                break
        sent1, sent2, sent3 = sent1[:i], sent2[:i], sent3[:i]
        # print('****************************************************************')
        # print(sent1)
        # print(sent2)
        # print(sent3)
        # print('****************************************************************')
        if not operator.eq(sent1, sent3):
            if operator.eq(sent2, sent3):
                TP_sent += 1
            else:
                FP_sent += 1
        else:
            if operator.eq(sent2, sent1):
                TN_sent += 1
            else:
                FN_sent += 1

        if len(sent1) == len(sent2) and len(sent1) == len(sent3):
            for i in range(len(sent1)):
                # if sent3[i] == 0:
                #     continue
                if sent1[i] != sent3[i]:
                    if sent3[i] == sent2[i]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if sent1[i] == sent2[i]:
                        TN += 1
                    else:
                        FN += 1
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN)  != 0 else 0
    precision = TP / (TP + FP)  if (TP + FP)  != 0 else 0
    recall = TP / (TP + FN)  if (TP + FN) != 0 else 0
    beta = 1
    F = (1 + beta ** 2) * precision * recall / ((beta ** 2 * precision) + recall) if ((beta ** 2 * precision) + recall) != 0 else 0
    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    F = round(F, 4)
    print('char level...')
    print("valid chars:\t{}\t\tacc:\t{}\t\tpre:\t{}\t\trecall:\t{}\t\tF:\t{}\t\t".format(TP + FP + TN + FN, accuracy, precision,
                                                                     recall, F))
    print("TP:\t{}\t\tFP:\t{}\\t\tfn:\t{}\t\tTN:\t{}".format(TP, FP, FN, TN))

    accuracy_sent = (TP_sent + TN_sent) / (TP_sent + FP_sent + FN_sent + TN_sent) if (TP_sent + FP_sent + FN_sent + TN_sent) != 0 else 0
    precision_sent = TP_sent / (TP_sent + FP_sent) if (TP_sent + FP_sent) != 0 else 0
    recall_sent = TP_sent / (TP_sent + FN_sent) if (TP_sent + FN_sent) != 0 else 0
    F_sent = (1 + beta ** 2) * precision_sent * recall_sent / ((beta ** 2 * precision_sent) + recall_sent) if ((beta ** 2 * precision_sent) + recall_sent) != 0 else 0
    accuracy_sent = round(accuracy_sent, 4)
    precision_sent = round(precision_sent, 4)
    recall_sent = round(recall_sent, 4)
    F_sent = round(F_sent, 4)

    print()
    print('sentence level...')
    print("valid sentences:\t{}\t\tacc:\t{}\t\tpre:\t{}\t\trecall:\t{}\t\tF:\t{}\t\t".format(TP_sent + FP_sent + TN_sent + FN_sent,
                                                                         accuracy_sent, precision_sent, recall_sent,
                                                                         F_sent))
    print("TP:\t{}\t\tFP:\t{}\t\tfn:\t{}\t\tTN:\t{}".format(TP_sent, FP_sent, FN_sent, TN_sent))
    return F_sent

