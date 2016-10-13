from lxml import etree

import OutService

class OutServiceXml(OutService.OutService):

    def printPredictions(self, predictions, ml_model="", sample_file="", *print_args, **print_kwargs ):
        """

        :param predictions: a list of pandas Timestamps
        :param print_args: as for normal print()
        :param print_kwargs: as for normal print()
        :return:
        """
        root  = etree.Element("ttc", encoding="UTF-8")
        preds = etree.SubElement(root, "predictions")
        if ml_model:
            preds.set("ml_model", ml_model)
        if sample_file:
            preds.set("sample_file", sample_file)

        for p in predictions:
            etree.SubElement(preds, "p").text = "%s" % p.isoformat()

        print(etree.tostring(root, pretty_print=True, encoding='unicode'), *print_args, **print_kwargs)