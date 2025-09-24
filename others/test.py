def removeDuplicates(nums):
    if len(nums) <= 2:
        return len(nums)
    
    slow = fast = 2
    
    while fast < len(nums):
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    
    return slow

# 示例测试
nums1 = [1, 1, 1, 2, 2, 3]
len1 = removeDuplicates(nums1)
result = [nums1[i] for i in range(len1)]
print(result)

def merge(nums1, m, nums2, n):
    # 初始化两个指针
    p1 = m - 1
    p2 = n - 1
    
    # 从后向前遍历，将较大的元素放到 nums1 的末尾
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p1 + p2 + 1] = nums1[p1]
            p1 -= 1
        else:
            nums1[p1 + p2 + 1] = nums2[p2]
            p2 -= 1
    
    # 如果 nums2 还有剩余元素，直接复制到 nums1 的前面
    if p2 >= 0:
        nums1[:p2 + 1] = nums2[:p2 + 1]

# 示例测试
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3
merge(nums1, m, nums2, n)
print(nums1)  # 输出: [1, 2, 2, 3, 5, 6]
# 时间复杂度：O(m+n)，其中 m 和 n 分别是 nums1 和 nums2 的长度。

def candy(ratings):
   n = len(ratings)
   if n == 1:
       return 1
   
   # 初始化每个孩子的糖果数为1
   candies = [1] * n
   
   # 第一次遍历：从左到右
   for i in range(1, n):
       if ratings[i] > ratings[i - 1]:
           candies[i] = candies[i - 1] + 1
   
   # 第二次遍历：从右到左
   for i in range(n - 2, -1, -1):
       if ratings[i] > ratings[i + 1]:
           candies[i] = max(candies[i], candies[i + 1] + 1)
   
   # 计算总共需要的糖果数
   return sum(candies)

# 示例测试
ratings1 = [1, 0, 2]
print(candy(ratings1))  # 输出：5

ratings2 = [1, 2, 2]
print(candy(ratings2))  # 输出：4