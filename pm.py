import tensorflow as tf

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# 00021ddc3.jpg,1
# 003fc760d.jpg,0
# 0056d301c.jpg,1
# 005883874.jpg,0
# 00804c4d8.jpg,0
# 0089fe989.jpg,0
# 00cd53ea8.jpg,0
# 00dc72cbc.jpg,0
# 00dea4e84.jpg,0
# 00f283a9b.jpg,0
# 00fd837c6.jpg,0
# 010ffc277.jpg,0
# 0123b84ee.jpg,1
# 0135aa332.jpg,0
# 015399ad1.jpg,0
# 018f45cf2.jpg,0
# 01d85d282.jpg,0
# 01ec512cd.jpg,0
# 0253c3406.jpg,1
# 02638d157.jpg,0
# 026cff18d.jpg,0
# 0289651ff.jpg,0
# 029919989.jpg,0
# 02ab1b029.jpg,0
# 02b8ad48c.jpg,1
# 02cddb0f3.jpg,1
# 02cfbf9e0.jpg,1
# 02fdaf33f.jpg,0
# 03618e3e7.jpg,0
# 036de67f5.jpg,0
# 036f9b076.jpg,0
# 037bc7abf.jpg,0
# 03b25bda9.jpg,1
# 03dd3138e.jpg,0
# 03e7e8903.jpg,0
# 0435ffc4b.jpg,0
# 045b32426.jpg,0
# 046c47bdc.jpg,0
# 04aaf1822.jpg,1
# 04ea43044.jpg,1
# 050224872.jpg,0
# 0534be6bb.jpg,1
# 05497d6ad.jpg,0
# 0563a62cc.jpg,0
# 057c01e33.jpg,0
# 0595c9425.jpg,0
# 05adb1913.jpg,0
# 05c008754.jpg,1
# 05c929e38.jpg,1
# 05ca7052b.jpg,0
