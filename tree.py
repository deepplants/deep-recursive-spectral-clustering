from typing import Optional, Tuple, List, Any, Dict, Literal, Union, Callable
from collections import ChainMap
from copy import deepcopy
import numpy as np
import logging

EPSILON = 1E-12

class TreeNode(object):
    """
    Represents a node in a tree structure.
    
    Attributes:
        _id (str): The ID of the node.
        _children (list): The list of child nodes.
        _parent (TreeNode): The parent node.
        _attrs (list): The list of attribute names passed as kwargs.
    
    Methods:
        id -> str: Returns the ID of the node.
        parent -> TreeNode: Returns the parent node.
        children -> list: Returns the list of child nodes.
        attrs -> list: Returns the list of attribute names.
        __hash__() -> int: Returns the hash value of the node.
        __eq__(other) -> bool: Checks if the node is equal to another node.
        __str__(depth=0, isLast=True, areAncestorsLast=None, showAttrs=None) -> str: Returns a string representation of the node.
        max_offspring() -> int: Returns the maximum number of offspring of the node.
        max_depth() -> int: Returns the maximum depth of the node.
        add_child(item) -> None: Adds a child node to the node.
        add_children(item) -> None: Adds multiple child nodes to the node.
        update_attribute(attr, map) -> None: Updates the value of an attribute in the node.
        exponential_indexing() -> dict: Returns a dictionary mapping node IDs to exponential indices.
        postorder_indexing() -> dict: Returns a dictionary mapping node IDs to postorder indices.
        nodes_list() -> list: Returns a list of all nodes in the tree.
        leaves_list() -> list: Returns a list of all leaf nodes in the tree.
        paths_list() -> list: Returns a list of all paths in the tree.
    """
    
    def __init__(
        self, 
        id: str = 'r',
        children: Optional[Union[List["TreeNode"], "TreeNode"]] = None,
        **kwargs
        ) -> None:
        self._id = id
        self._children = list()
        self._parent = None
        self.add_children(children)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._attrs = list(kwargs.keys())
    
    @property
    def id(self):
        return self._id
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def children(self):
        return self._children

    @property
    def attrs(self):
        return self._attrs

    @id.setter
    def id(self, new_id):
        self._id = new_id
    
    @parent.setter
    def parent(self, new_parent):
        self._parent = new_parent
        
    @children.setter
    def children(self, new_children):
        self._children = new_children
    
    @attrs.setter
    def attrs(self, new_attrs):
        self._attrs = new_attrs
    
    def ancestor(self, generations: int) -> "TreeNode":
        return self.dinasty()[::-1][generations]
    
    def dinasty(self) -> List["TreeNode"]:
        return (self._parent.dinasty() if self._parent is not None else []) + [self]
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
        
    def __hash__(self):
        # return hash(self.__str__())
        return (str(self.id)+str(self.attrs)+('').join(str(getattr(self,attr)) for attr in self.attrs)).__hash__()
    
    def __eq__(
        self, 
        other
        ) -> bool:
        if not isinstance(other, TreeNode):
            return False
        elif len(self.attrs) != len(other.attrs):
            return False
        elif not all([a in other.attrs for a in self.attrs]):
            return False
        return (self._id == other.id and
                # len(self.children) == len(other.children) and
                # all([i in self.children for i in other.children]) and
                all([getattr(self,a)==getattr(other,a) for a in self.attrs]))
        
    def __str__(
        self, 
        depth: int = 0, 
        isLast: bool = True, 
        areAncestorsLast: Optional[List[bool]] = None,
        showAttrs: Optional[List[str]] = None
    ) -> str:  
        """
        Returns a string representation of the tree node and its children.

        Args:
            depth (int): The depth of the current node in the tree (default: 0).
            isLast (bool): Indicates whether the current node is the last child of its parent (default: True).
            areAncestorsLast (Optional[List[bool]]): A list of booleans indicating whether the ancestors of the 
            current node are the last children of their parents (default: None).
            showAttrs (Optional[List[str]]): A list of attribute names to include in the string representation (default: None).

        Returns:
            str: A string representation of the tree node and its children.
        """
        
        if areAncestorsLast is None:
            areAncestorsLast = [] 
        mess = ""   
        for i in range(1, depth):
            if not areAncestorsLast[i]:
                mess +=  "|   "
            else:
                mess +=  "    "
        attrs = [f"{att}:{getattr(self, att)}" for att in showAttrs if hasattr(self, att)] \
            if showAttrs is not None else [f"{att}:{getattr(self, att)}" for att in self._attrs]
        if depth == 0:
            mess += str(self._id) + "\t" + "\t".join(attrs) + "\n"
        else:
            mess += "+---" + str(self._id) + "\t" + "\t".join(attrs) + "\n"
        areAncestorsLast.append(isLast)
        for i, child in enumerate(self.children):
            mess += child.__str__(
                depth + 1, 
                i == (len(self.children) - 1), 
                deepcopy(areAncestorsLast),
                showAttrs
            )

        return mess

    def max_offspring(
        self,
        ) -> int:
        if not self.is_leaf():
            return max(len(self.children), *[child.max_offspring() for child in self.children])
        return 0
    
    def max_depth(
        self,
        ) -> int:
        if not self.is_leaf():
            return 1 + max([child.max_depth() for child in self.children])
        return 0
        
    def add_child(
        self, 
        item: Optional["TreeNode"],
        override_id: bool = True
        ) -> None:
        if item is not None:
            if item not in self.children:
                item._parent = self
                if override_id:
                    if item.is_leaf():
                        item._id = self._id + 'c' + str(len(self.children))
                    else:
                        def rename(node, old, new):
                            setattr(node, "_id", node.id.replace(old, new))
                            for child in node.children:
                                rename(child, old, new)
                        new = self._id + 'c' + str(len(self.children))
                        old = item.id
                        rename(item, old, new)
                      
                self._children.append(item)
            else:
                raise ValueError('Item present yet.')
    
    def add_children(
        self, 
        item: Optional[Union[List["TreeNode"], "TreeNode"]]
        ) -> None:
        if isinstance(item, list):
            for i in item:
                self.add_child(i)
        elif isinstance(item, TreeNode):
            self.add_child(item)
        elif item is None:
            pass
        else:
            raise ValueError("item type not supported")
             
    def update_attribute(
        self,
        attr: str,
        mapping: Dict["TreeNode",Any]
        ) -> None:
        """
        Update the attribute of the current node with the corresponding value from the given map.

        Args:
            attr (TreeNode): The name of the attribute to update.
            map (Dict[TreeNode,Any]): A dictionary mapping nodes to attribute values.

        Returns:
            None
        """
        if not self in mapping:
            logging.debug("Node id not present in current mapping, skipping..")
        # elif not hasattr(self, attr):
        #     logging.debug("Attribute not present in current node, skipping..")
        else:
            setattr(self, attr, mapping[self])
        
        for child in self.children:
            child.update_attribute(attr, mapping)
            
    def exponential_indexing(
        self
        ) -> Dict["TreeNode",int]:
        """
        Perform exponential indexing on the tree nodes.

        Returns:
            A dictionary mapping nodes to their corresponding index values.
        """
        depth = self.max_depth()
        max_offspring = self.max_offspring()
        
        def func(node: "TreeNode", count: int, depth: int, basis: int) -> Dict["TreeNode",int]:
            return dict(ChainMap(*[{node:count}]+[func(child, count + index * basis ** (depth-1), depth-1, basis) for index, child in enumerate(node.children)]))
            
        return func(node=self, count=0, depth=depth, basis=max_offspring)

    def postorder_indexing(self, start=0) -> Dict["TreeNode", int]:
        """
        Performs a postorder traversal of the tree and returns a dictionary mapping each node's ID to its index.

        Returns:
            A dictionary mapping each node's ID to its index.

        Example:
            tree = TreeNode()
            # ... add nodes to the tree ...
            indexing = tree.postorder_indexing()
            print(indexing)
            # Output: {'node1.id': 0, 'node2.id': 1, 'node3.id': 2, ...}
        """
        def func(node: "TreeNode", count: int, mapping: dict) -> Tuple[int, Dict["TreeNode", int]]:
            for child in node.children:
                count, mapping = func(child, count, mapping)
            mapping[node] = count
            return count + 1, mapping

        return func(node=self, count=start, mapping=dict())[-1]
    
    def preorder_indexing(self) -> Dict["TreeNode", int]:
        """
        Performs a preorder traversal of the tree and returns a dictionary mapping each node's ID to its index.

        Returns:
            A dictionary mapping each node's ID to its index.

        Example:
            tree = TreeNode()
            # ... add nodes to the tree ...
            indexing = tree.preorder_indexing()
            print(indexing)
            # Output: {'node1.id': 0, 'node2.id': 1, 'node3.id': 2, ...}
        """
        def func(node: "TreeNode", count: int, mapping: dict) -> Tuple[int, Dict["TreeNode", int]]:
            mapping[node] = count
            for child in node.children:
                count, mapping = func(child, count+1, mapping)
            return count, mapping

        return func(node=self, count=0, mapping=dict())[-1]
    
    def nodes_list(
        self,
        ) -> List["TreeNode"]:        
        return sum([child.nodes_list() for child in self.children],[self])
    
    def cardinality(
        self
        ) -> int:
        return len(self.nodes_list())
    
    def leaves_list(
        self,
        ) -> List["TreeNode"]:        
        return [self] if self.is_leaf() else sum([child.leaves_list() for child in self.children],[])
    
    def paths_list(
        self,
        ) -> List[List["TreeNode"]]:
        leaves = self.leaves_list()
        return [leaf.dinasty() for leaf in leaves]

    def depths_dict(
        self,
        depth: int = 0,
        ) -> Dict["TreeNode", int]:
        
        return dict(ChainMap(*[{self:depth}]+[child.depths_dict(depth+1) for child in self.children])) if not self.is_leaf() else {self:depth} # type: ignore
    
    def to_linked_list(
        self
        ) -> Dict["TreeNode",List["TreeNode"]]:
        """
        Converts the tree structure into a linked list representation.

        Returns:
            A dictionary where each key is a TreeNode object and the corresponding value is a list of its children.
        """
        return dict(ChainMap(*[{self:[c for c in self.children]}]+[c.to_linked_list() for c in self.children])) # type: ignore
    
    def to_adjacency(
        self
        ) -> Tuple[Dict["TreeNode",int], np.ndarray]:
        """
        Converts the tree to an adjacency matrix representation.

        Returns:
            A tuple containing a dictionary mapping tree nodes to matrix indices and
            a numpy array representing the adjacency matrix.
        """
        linked_list = self.to_linked_list()
        size = len(linked_list)
        mapp = dict(zip(linked_list.keys(),range(size)))
        adj = np.full((size,size), False)
        for k, lst in linked_list.items():
            for v in lst:
                adj[mapp[k],mapp[v]] = True
        return mapp, adj

    def to_newick(self):
        
        if not self.is_leaf():
            return f'({",".join([child.to_newick() for child in self.children])})'
        return str(self.id)
        
    # def prune(
    #     self
    #     ) -> "TreeNode":
    #     """
    #     Prunes the current node from the tree.

    #     Raises:
    #         ValueError: If the current node is the root node or if each parent node in the tree
    #                     does not have at least two children.
    #     """
    #     # def rename(node, old, new):
    #     #     setattr(node, "_id", node.id.replace(old, new))
    #     #     for child in node.children:
    #     #         rename(child, old, new)

    #     if self.parent is None:
    #         pass
    #     elif len(self.parent.children) == 1:
    #         self.parent._children.remove(self)
    #         # for attr in self.parent.attrs:
    #         #     delattr(self.parent, attr)
    #         # for attr in self.attrs:
    #         #     setattr(self.parent, attr, getattr(self, attr))
    #         # self.parent._attrs = self.attrs
    #         # self.parent._children = []
    #     elif len(self.parent.children) == 2:
    #         # print(self.parent.name, self.parent.id)
    #         # print(self.name, self.id)
    #         index = self.parent.children.index(self)
    #         # self.parent._children[index]._parent = None
    #         other = self.parent.children[1-index]
    #         # self.parent._children[1-index]._parent = None
    #         self.parent._children.remove(self)
    #         for attr in self.parent.attrs:
    #             delattr(self.parent, attr)
    #         for attr in other.attrs:
    #             setattr(self.parent, attr, getattr(other, attr))
    #         self.parent._attrs = other.attrs
    #         self.parent._children = other.children
    #         self.parent._id = other.id
    #         other._parent = None
    #         # rename(self.parent, other.id, self.parent.id)
    #     elif len(self.parent.children) > 2:
    #         index = self.parent.children.index(self)
    #         self.parent._children.remove(self)
    #         # idnew = self.id
    #         # for index, sibs in enumerate(self.parent.children[index:]):
    #         #     temp = str(sibs.id)
    #         #     rename(sibs, sibs.id, idnew)
    #         #     idnew = temp            
    #     else:
    #         raise ValueError(f'each parent node in a tree must have at least one child, got {len(self.parent.children)}.')
    #     self._parent = None
    #     return self
    
    def prune(
        self
        ) -> "TreeNode":
        """
        Prunes the current node from the tree.

        Raises:
            ValueError: If the current node is the root node or if each parent node in the tree
                        does not have at least two children.
        """
        # def rename(node, old, new):
        #     setattr(node, "_id", node.id.replace(old, new))
        #     for child in node.children:
        #         rename(child, old, new)
        if self.parent is None:
            raise ValueError("Can't prune root node.")
        # index = self.parent.children.index(self)
        self.parent._children.remove(self)
        # idnew = self.id
        # for index, sibs in enumerate(self.parent.children[index:]):
        #     temp = str(sibs.id)
        #     rename(sibs, sibs.id, idnew)
        #     idnew = temp    
        self._parent = None
        return self
    

def shortest_path(
    node_a : TreeNode,
    node_b : TreeNode,
    ) -> List[TreeNode]:
    """
    Finds the shortest path between two nodes in a tree.

    Args:
        node_a (TreeNode): The first node.
        node_b (TreeNode): The second node.
        root (TreeNode, optional): The root of the tree. Defaults to None.
        paths (List[List[TreeNode]], optional): The list of all paths in the tree. Defaults to None.

    Returns:
        List[TreeNode]: The shortest path between node_a and node_b as a list of nodes.

    Raises:
        AssertionError: If node_a or node_b is not present in any of the paths.

    """
    if node_a == node_b:
        return [node_a]
    
    path_to_a = node_a.dinasty()
    path_to_b = node_b.dinasty()
        
    # latest common ancestor
    common_ancestors = [x_a for x_a,x_b in zip(path_to_a,path_to_b) if x_a==x_b]
    
    assert len(common_ancestors) > 0, "No common ancestor found."
    
    lca = common_ancestors[-1]
    
    return  path_to_a[path_to_a.index(lca):path_to_a.index(node_a)+1][::-1] + \
        path_to_b[path_to_b.index(lca)+1:path_to_b.index(node_b)+1]
        
def search(
    node: TreeNode,
    attr: str,
    value: Any,
    comp: Callable[[Any,Any],bool] = (lambda x,y: x==y)
    ) -> List[TreeNode]:
    """
    Performs a depth-first search on the tree to find nodes that have a specific attribute value.

    Args:
        attr (str): The name of the attribute to search for.
        value (Any): The value of the attribute to match.

    Returns:
        List[TreeNode]: A list of tree nodes that have the specified attribute value.
    """
    return sum([search(child, attr, value, comp) for child in node.children], [node] if comp(getattr(node, attr, None), value) else [])
    
def subtree_from_nodes_list(
    nodes_list: List[TreeNode]
    ) -> TreeNode:
    """
    Returns a subtree that only contains the nodes present in the nodes_list and the inner nodes.

    Args:
        node (TreeNode): The root node of the tree.
        nodes_list (List[TreeNode]): The list of nodes to be included in the subtree.

    Returns:
        TreeNode: The root node of the subtree, or None if no subtree can be formed.

    """
    assert len(nodes_list) > 0, "Empty list of nodes."
    nodes_list = deepcopy(nodes_list)
    paths_to_nodes_in_list = [leaf.dinasty() for leaf in nodes_list]
    assert all([paths_to_nodes_in_list[0][0] == path[0] for path in paths_to_nodes_in_list]), "Nodes do not share a common root."
    root = paths_to_nodes_in_list[0][0]
    depths = root.depths_dict()
    all_nodes = set(root.nodes_list())
    keep_nodes = set(sum(paths_to_nodes_in_list,[]))
    node_to_delete = sorted(list(all_nodes.difference(keep_nodes)),key=lambda x: depths[x], reverse=True)
    for node in node_to_delete:
        node.prune()
    return root

def mask_node_leaves_in_values(node: TreeNode, attr: str, values: np.ndarray) -> np.ndarray:
    """
    Masks the leaves of a given node in an array of values based on a specified attribute.

    Args:
        node (TreeNode): The node whose leaves should be masked.
        attr (str): The attribute to compare against the values.
        values (np.ndarray): The array of values to be masked.

    Returns:
        np.ndarray: A boolean mask indicating which values correspond to the leaves of the node.
    """
    mask = np.full(values.shape, False)
    mask[np.isin(values, [getattr(leaf, attr) for leaf in node.nodes_list() if hasattr(leaf, attr)])] = True
    return mask
    
def mask_subtree_nodes_leaves_in_values(
    root: TreeNode,
    attr: str,
    values: np.ndarray
    ) -> Tuple[Dict[int,TreeNode],np.ndarray]:
    nodes_list = root.nodes_list()
    nodes_list.remove(root)
    masks = []
    for node in nodes_list:
        masks.append(mask_node_leaves_in_values(node, attr, values))
    mapp = dict(zip(range(len(nodes_list)),nodes_list))
    masks = np.stack(masks,axis=0)
    return mapp, masks

def mask_subtree_nodes_leaves_in_values_at_depth(
    depth: int,
    root: TreeNode,
    attr: str,
    values: np.ndarray
    ) -> Tuple[Dict[int,TreeNode],np.ndarray]:
    assert depth>0, 'depth must be > 0'
    nodes_list = [k for k,v in root.depths_dict().items() if v==depth]
    if root in nodes_list: nodes_list.remove(root)
    masks = []
    for node in nodes_list:
        masks.append(mask_node_leaves_in_values(node, attr, values))
    mapp = dict(zip(range(len(nodes_list)),nodes_list))
    masks = np.stack(masks,axis=0)
    return mapp, masks

def pairwise_unoriented_distances(
    root: TreeNode,
    leaves_only: bool = False
    ) -> Tuple[Dict[int,TreeNode], Dict[Tuple[int,int],int]]:
    mapp, pair_to_path = pairwise_unoriented_paths(root, leaves_only)
    pair_to_dist = dict()
    for pair, path in pair_to_path.items():
        pair_to_dist[pair] = len(path)-1
    return mapp, pair_to_dist
    
def pairwise_unoriented_paths(
    root: TreeNode,
    leaves_only: bool = False
    ) -> Tuple[Dict[int,TreeNode], Dict[Tuple[int,int],List[int]]]:
    nodes_list = root.nodes_list() if not leaves_only else root.leaves_list()
    num_nodes = len(nodes_list)
    mapp = dict(zip(range(num_nodes), nodes_list))
    unique_pairs = [(i,j) for i in range(num_nodes) for j in range(num_nodes)]
    pair_to_path = dict()
    for pair in unique_pairs:
        pair_to_path[pair] = shortest_path(mapp[pair[0]], mapp[pair[1]])
    return mapp, pair_to_path
    
def pairwise_oriented_reachability(
    root: TreeNode,
    ) -> Tuple[Dict[int,TreeNode], Dict[Tuple[int,int],bool]]:
    nodes_list = root.nodes_list()
    num_nodes = len(nodes_list)
    mapp = dict(zip(range(num_nodes), nodes_list))
    unique_pairs = [(i,j) for i in range(num_nodes) for j in range(num_nodes)]
    pair_to_reach = dict()
    for pair in unique_pairs:
        pair_to_reach[pair] = bool(mapp[pair[0]] in mapp[pair[1]].dinasty())
    return mapp, pair_to_reach
    
def pool_embeddings(
    root: TreeNode,
    points: np.ndarray, 
    attr: str,
    values: np.ndarray, 
    pooling: Literal['average','maximum','attention'] = 'average',
    mapping: Dict[Any, np.ndarray] = dict(),
    temp: float = 0.9,
    leaves_only: bool = True
    ) -> Dict[Any, np.ndarray]:
    for child in root.children:
        mapping = pool_embeddings(child, points, attr, values, pooling, mapping, temp, leaves_only)
    mask = mask_node_leaves_in_values(root, attr, values)
    if pooling == 'average':
        emb = np.mean(points[mask],axis=0,keepdims=False)
    elif pooling == 'median':
        emb = np.median(points[mask],axis=0,keepdims=False)
    elif pooling == 'maximum':
        emb = np.max(points[mask],axis=0,keepdims=False)
    elif pooling == 'attention':
        def softmax_stable(x):
            return np.exp(x - np.max(x,axis=-1,keepdims=True)) / np.maximum(np.exp(x - np.max(x,axis=-1,keepdims=True)).sum(axis=-1,keepdims=True),EPSILON)
        pts = points[mask]
        embs = softmax_stable((pts @ pts.T) / temp) @ pts
        emb = np.mean(embs,axis=0,keepdims=False)
    if hasattr(root,attr) and np.any(mask):
        if leaves_only and root.is_leaf():
            mapping[getattr(root,attr)] = l2_normalization(emb)
        elif not leaves_only:
            mapping[getattr(root,attr)] = l2_normalization(emb)
    return mapping

def normalization(a: np.ndarray, p: int = 2, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    return a / np.maximum(np.sum(a**p,axis=axis,keepdims=True)**(1./p),EPSILON)


def l2_normalization(a: np.ndarray, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    return normalization(a, 2, axis)

def right_relatives_dict(root: TreeNode) -> Dict["TreeNode",List["TreeNode"]]:
    """
    Returns a dictionary mapping each node in the tree to a list of its right relatives.

    Args:
        root (TreeNode): The root node of the tree.

    Returns:
        Dict[TreeNode, List[TreeNode]]: A dictionary mapping each node to a list of its right relatives.
    """
    node_to_post_idx = root.postorder_indexing()
    node_to_pre_idx = root.preorder_indexing()
    all_nodes = root.nodes_list()
    mapping = dict()
    for node in all_nodes:
        mapping[node] = [x for x in all_nodes if node_to_pre_idx[node] < node_to_pre_idx[x] and node_to_post_idx[node] < node_to_post_idx[x]]
    
    return mapping
    
def left_relatives_dict(root: TreeNode) -> Dict["TreeNode",List["TreeNode"]]:
    """
    Returns a dictionary mapping each node in the tree to a list of its left relatives.

    Args:
        root (TreeNode): The root node of the tree.

    Returns:
        Dict[TreeNode, List[TreeNode]]: A dictionary mapping each node to a list of its left relatives.
    """
    node_to_post_idx = root.postorder_indexing()
    node_to_pre_idx = root.preorder_indexing()
    all_nodes = root.nodes_list()
    mapping = dict()
    for node in all_nodes:
        mapping[node] = [x for x in all_nodes if node_to_pre_idx[node] > node_to_pre_idx[x] and node_to_post_idx[node] > node_to_post_idx[x]]
    
    return mapping
    
def descendants_dict(root: TreeNode) -> Dict[TreeNode,List[TreeNode]]:
    """
    Returns a dictionary mapping each node in the tree to its list of descendants.

    Args:
        root (TreeNode): The root node of the tree.

    Returns:
        Dict[TreeNode, List[TreeNode]]: A dictionary mapping each node to its list of descendants.
    """
    all_nodes = root.nodes_list()
    mapping = dict()
    for node in all_nodes:
        mapping[node] = node.nodes_list()
        mapping[node].remove(node)
        
    return mapping

def children_dict(root: TreeNode) -> Dict[TreeNode,List[TreeNode]]:
    """
    Returns a dictionary mapping each node in the tree to its list of descendants.

    Args:
        root (TreeNode): The root node of the tree.

    Returns:
        Dict[TreeNode, List[TreeNode]]: A dictionary mapping each node to its list of descendants.
    """
    all_nodes = root.nodes_list()
    mapping = dict()
    for node in all_nodes:
        mapping[node] = node.children
        
    return mapping

def find_embedding(s: TreeNode, t: TreeNode, comp: Callable) -> Dict[Tuple[int,int],int]:
    """
    Algorithm taken from "The tree inclusion problem" by Pekka Kilpelainen Heildd Mannila
    """    
    # get postorder mapping {TreeNode: Postorder Index}
    s_node_to_idx = s.postorder_indexing(start=1)
    t_node_to_idx = t.postorder_indexing(start=1)
    
    # reverse the mapping
    s_idx_to_node = dict((v,k) for k,v in s_node_to_idx.items())
    t_idx_to_node = dict((v,k) for k,v in t_node_to_idx.items())
    
    # sort the dict by index
    s_idx_to_node = dict(sorted(s_idx_to_node.items()))
    t_idx_to_node = dict(sorted(t_idx_to_node.items()))
    
    m = max(s_idx_to_node.keys())
    n = max(t_idx_to_node.keys())
    
    logging.debug(f"m={m}, n={n}")
    # init embedding table
    INF = 1e12
    e = dict(((u,v),int(INF)) for v in range(0,n) for u in range(1,m+1))
    logging.debug(f"table e = {e}")
    # children dict
    childs_dict = children_dict(s)
    childs_dict_idx = dict((s_node_to_idx[k],list(s_node_to_idx[vv] for vv in v)) for k,v in childs_dict.items())
    logging.debug(f"s nodes children = {childs_dict_idx}")

    # descendants dict
    desc_dict = descendants_dict(t)
    desc_dict_idx = dict((t_node_to_idx[k],list(t_node_to_idx[vv] for vv in v)) for k,v in desc_dict.items())
    logging.debug(f"t nodes descendants = {desc_dict_idx}")

    # left relatives dict
    lr_dict = left_relatives_dict(t)
    lr_dict_idx = dict((t_node_to_idx[k],list(t_node_to_idx[vv] for vv in v)) for k,v in lr_dict.items())
    logging.debug(f"t nodes left relatives = {lr_dict_idx}")
    
    for u, u_node in s_idx_to_node.items():
        logging.debug(f"table e = {e}")
        logging.debug(f"looking s node u={u}")
        # breakpoint()
        u_k = dict((i+1,kk) for i,kk in enumerate(childs_dict_idx[u]))
        q = 0
        k = max(u_k.keys()) if len(u_k) else 0
        logging.debug(f"k={k}")
        for v, v_node in t_idx_to_node.items():
            logging.debug(f"looking t node v={v}")
            # breakpoint()
            if comp(u_node,v_node):
                logging.debug(f"found same label u={u} and v={v}")
                p = min(desc_dict_idx[v]+[int(INF)])-1
                i = 0
                logging.debug(f"p={p}")
            
                while i < k and p < v:
                    p = e[(u_k[i+1],p)] # type: ignore
                    logging.debug(f"p={p}")
                    if p in desc_dict_idx[v]: i += 1
                
                if i == k:
                    logging.debug(f"u={u} and lr(u)={lr_dict_idx[v]}")
                    while q in lr_dict_idx[v]+[0]: # 0 is left relative of anyone
                        e[(u,q)] = v
                        q += 1
                        logging.debug(f"q={q}")
                        logging.debug(f"table e updated e={e}")
                    
    return e
