$$\text{w-len} = l_1.l_2 + l_2.l_3 + ... + l_{n-1}.l_n$$

$$\text{b-len} = l_2 + l_3 + ... + l_n$$

$$z_{l_2} = f_2 \left[ \sum_{l_1=0}^{L_1-1} (x_{l_1}.w_{l_1 + L_1l_2}) + b_{l_2} \right]$$

$$z_{l_3} = f_3 \left[ \sum_{l_2=0}^{L_2-1} (z_{l_2}.w_{l_2 + L_1L_2 + L_2 l_3}) + b_{L_2 + l_3} \right]$$

$$z_{l_4} = f_4 \left[ \sum_{l_3=0}^{L_3-1} (z_{l_3}.w_{l_3 + L_1L_2 + L_2 L_3 + L_3 l_4}) + b_{L_2 + L_3 + l_4} \right]$$

$$\vdots$$

$$z_{l_n} = f_n \left[ \sum_{l_{n-1}=0}^{L_{n-1}-1} (z_{l_{n-1}}.w_{l_{n-1} + L_1L_2 + L_2 L_3 + ... + L_{n-2} L_{n-1} + L_{n-1} l_n}) + b_{L_2 + L_3 + ...+ L_{n-1} +l_n} \right]$$

$$l_{n-1} + L_1L_2 + L_2 L_3 + ... + L_{n-2} L_{n-1} + L_{n-1} l_n = l_{n-1} + \sum_{r=2}^{n-1} L_{r-1} L_{r} + L_{n-1}l_n$$

$$L_2 + L_3 + ...+ L_{n-1} +l_n = \sum_{r=2}^{n-1}L_r + l_n$$

$$z_{l_n} = f_n \left[ \sum_{l_{n-1}=0}^{L_{n-1}-1} (z_{l_{n-1}}.w_{l_{n-1} + \sum_{r=2}^{n-1} L_{r-1} L_{r} + L_{n-1}l_n}) + b_{\sum_{r=2}^{n-1}L_r + l_n} \right]$$

$$loss = \frac{1}{L_n} \sum_{l_n=0}^{L_n-1} (y_{l_n} - z_{l_n})^2$$

# grad

$$\frac{\partial(loss)}{\partial(z_{l_n})} = -2(y_{l_n} - z_{l_n})$$

$$\frac{\partial(z_{l_n})}{\partial(z_{l_{n-1}})} =f_n' \left[ \sum_{l_{n-1}=0}^{L_{n-1}-1} (z_{l_{n-1}}.w_{l_{n-1} + \sum_{r=2}^{n-1} L_{r-1} L_{r} + L_{n-1}l_n}) + b_{\sum_{r=2}^{n-1}L_r + l_n} \right].w_{l_{n-1} + \sum_{r=2}^{n-1} L_{r-1} L_{r} + L_{n-1}l_n}$$

$$\frac{\partial z_{l_4}}{\partial z_{l_3}} = f_4' \left[ \sum_{l_3=0}^{L_3-1} (z_{l_3}.w_{l_3 + L_1L_2 + L_2 L_3 + L_3 l_4}) + b_{L_2 + L_3 + l_4} \right].w_{l_3 + L_1L_2 + L_2 L_3 + L_3 l_4}$$

$$\frac{\partial z_{l_3}}{\partial z_{l_2}} = f_3' \left[ \sum_{l_2=0}^{L_2-1} (z_{l_2}.w_{l_2 + L_1L_2 + L_2 l_3}) + b_{L_2 + l_3} \right].w_{l_2 + L_1L_2 + L_2 l_3}$$

$$\frac{\partial z_{l_2}}{\partial w_{l_1}} = f_2' \left[ \sum_{l_1=0}^{L_1-1} (x_{l_1}.w_{l_1 + L_1l_2}) + b_{l_2} \right].x_{l_1}$$

$$\frac{d(loss)}{dw_{l_1}} = \frac{\partial(loss)}{\partial(z_{l_n})} . \frac{\partial(z_{l_n})}{\partial(z_{l_{n-1}})} \dots \frac{\partial z_{l_4}}{\partial z_{l_3}} . \frac{\partial z_{l_3}}{\partial z_{l_2}} . \frac{\partial z_{l_2}}{\partial w_{l_1}}$$
