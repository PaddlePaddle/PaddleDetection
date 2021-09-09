import paddle
import math

def wing_loss(landmarks, labels, w=0.06, epsilon=0.01):
        """
        Arguments:
            landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
            w, epsilon: a float numbers.
        Returns:
            a float tensor with shape [].
        """

        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = paddle.abs(x)

        losses = paddle.where(\
        (w>absolute_x),\
        w * paddle.log(1.0 + absolute_x / epsilon),\
        absolute_x - c)


        # loss = tf.reduce_mean(tf.reduce_mean(losses, axis=[1]), axis=0)
        losses = paddle.mean(losses,axis=1,keepdim=True)
        loss = paddle.mean(losses)
        return loss

def got_total_wing_loss(output,crop_landmarks):
    loss = wing_loss(output, crop_landmarks)
    return loss
