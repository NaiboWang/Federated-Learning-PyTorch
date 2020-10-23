"""
The following Python implementation of Shamir's Secret Sharing is
released into the Public Domain under the terms of CC0 and OWFa:
https://creativecommons.org/publicdomain/zero/1.0/
http://www.openwebfoundation.org/legal/the-owf-1-0-agreements/owfa-1-0

See the bottom few lines for usage. Tested on Python 2 and 3.
"""

from __future__ import division
from __future__ import print_function

import random
import functools

# 12th Mersenne Prime 梅森素数
# (for this application we want a known prime number as close as
# possible to our security level; e.g.  desired security level of 128
# bits -- too large and all the ciphertext is large; too small and
# security is compromised)
_PRIME = 2 ** 127 - 1
# 13th Mersenne Prime is 2**521 - 1

# python标准库--functools.partial_随风的山羊的博客-CSDN博客
# https://blog.csdn.net/Zjack_understands/article/details/80242946
_RINT = functools.partial(random.SystemRandom().randint, 0) # 返回一个从0到指定参数的随机数

def _eval_at(poly, x, prime):
    """Evaluates polynomial (coefficient tuple) at x, used to generate a
    shamir pool in make_random_shares below.
    这一步即得到了f(x) = a0+a1x+a2x^2+a3x^3+...+(ak-1)x^(k-1) % p 的值
    """
    accum = 0
    for coeff in reversed(poly):
        accum *= x
        accum += coeff
        accum %= prime
    return accum

def make_random_shares(minimum, shares, prime=_PRIME,secret=None):
    """
    Generates a random shamir pool, returns the secret and the share
    points.
    """
    if minimum > shares:
        raise ValueError("Pool secret would be irrecoverable.")
    poly = [_RINT(prime - 1) for i in range(minimum)] # 生成minimum个随机数，每个随机数的范围为[0,prime-1],其中，poly[0]为secret，其余minimum-1项为多项式系数
    if secret != None:
        poly[0] = secret # 手动指定secret值
    points = [(i, _eval_at(poly, i, prime))
              for i in range(1, shares + 1)]
    return poly[0], points

def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1) this can
    be computed via extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation

    扩展欧几里得算法 - 维基百科，自由的百科全书
    https://zh.wikipedia.org/wiki/%E6%89%A9%E5%B1%95%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95

    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b # 3//2=1,a是den的值，b是prime
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    # 最后一步 b=0时，a就是最大公约数
    # 扩展欧几里得算法 - 维基百科，自由的百科全书
    # https://zh.wikipedia.org/wiki/%E6%89%A9%E5%B1%95%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95
    # 这里的last_x和last_y有如下特性：last_x * a + last_y * b = gcd(a,b)
    # 如果a和b互为素数,则gcd(a,b) = 1, 这里的b即p,因此有last_x * a = gcd(a,b) - last_y * b = 1 - last_y * p,
    # 则(last_x * a) % p = (1 - last_y * p) % p = 1,即last_x即a关于b的乘法逆元！！
    return last_x, last_y

def _divmod(num, den, p):
    """Compute num / den modulo prime p

    To explain what this means, the return value will be such that
    the following is true: den * _divmod(num, den, p) % p == num
    """
    # inv即den模p的乘法逆元,有(den*inv)%p=1,inv=(pt+1)/den，p/t/den均为整数且结果inv为整数
    inv, _ = _extended_gcd(den, p)
    return num * inv # 结果为(pt+1) * num / den 对p求模之后得到结果为num/den 且 den * _divmod(num, den, p) % p == num

def PI(vals):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        accum *= v
    return accum

# interpolate插入，篡改
def _lagrange_interpolate(x, x_s, y_s, p):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    这里x=0即求f(0)即secret的值
    使用的是wiki上的这个部分的公式：
    Computationally efficient approach
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct" # 不相同的时候触发断言
    nums = []  # avoid inexact division
    dens = []
    for i in range(k):
        others = list(x_s)
        cur = others.pop(i) # pop取出值并删掉原数组内的元素
        nums.append(PI(x - o for o in others)) # nums: [x1x2,x0x2,x0x1]
        dens.append(PI(cur - o for o in others)) # dens: [(x0-x1)(x0-x2),(x1-x0)(x1-x2),(x2-x0)(x2-x1)]
    den = PI(dens) # den:(x0-x1)(x0-x2)(x1-x0)(x1-x2)(x2-x0)(x2-x1)
    num = sum([_divmod(nums[i] * den * y_s[i] % p, dens[i], p)
               for i in range(k)]) # num: [(x1-x0)(x1-x2)(x2-x0)(x2-x1) * f(x0) * x1x2  + (x0-x1)(x0-x2)(x2-x0)(x2-x1) * f(x1)*x0x2 + (x0-x1)(x0-x2)(x1-x0)(x1-x2) * f(x2) * x0x1] % p
    return (_divmod(num, den, p) + p) % p # 结果为num/den,即wiki上的公式

def recover_secret(shares, prime=_PRIME):
    """
    Recover the secret from share points
    (x, y points on the polynomial).
    """
    if len(shares) < 2:
        raise ValueError("need at least two shares")
    # Python zip() 函数 | 菜鸟教程
    # https://www.runoob.com/python/python-func-zip.html
    x_s, y_s = zip(*shares) # x_s:(1,2,3)即不同share的x y_s:(f(x1),f(x2),f(x3))
    return _lagrange_interpolate(0, x_s, y_s, prime)

def main(m=3,ns=6,sec=789):
    """Main function"""
    secret, shares = make_random_shares(minimum=m, shares=ns,secret=sec)

    print('Secret:                                                     ',
          secret)
    print('Shares:')
    if shares:
        for share in shares:
            print('  ', share)

    print('Secret recovered from minimum subset of shares:             ',
          recover_secret(shares[:m]))
    print('Secret recovered from a different minimum subset of shares: ',
          recover_secret(shares[-m:]))

if __name__ == '__main__':
    main(2,6,789789)
    a,b = _extended_gcd(5,3)
    print(a,b)