# Tensorflow使用load_variable加载模型
### tf.get_collection
```python
tf.get_collection(
    key,
    scope=None
)

```
封装在`Graph.get_collection()`使用默认图.

更多详情查看`tf.Graph.get_collection()`
```
Args:

    key: The key for the collection. For example, the GraphKeys class contains many standard names for collections.
    scope: (Optional.) If supplied, the resulting list is filtered to include only items whose name attribute matches using re.match. Items without a name attribute are never returned if a scope is supplied and the choice or re.match means that a scope without special tokens filters by prefix.

Returns:

    The list of values in the collection with the given name, 
    or an empty list if no value has been added to that collection. 
    The list contains the values in the order under which they were collected.
```
### 使用
