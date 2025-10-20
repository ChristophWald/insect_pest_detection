#from data/base.py, class BaseDataset

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        label = self.get_image_and_label(index)  
        weight = label.get("weight", None)
        
        item = self.transforms(label)
        if weight is not None:
            item["weight"] = weight
        #print("Test for flattening.")
        #print(item["weight"])
        return item