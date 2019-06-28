'''
--------------------------------------------------------------------------------
Quaternion.py
--------------------------------------------------------------------------------
Quaternion objects and methods
--------------------------------------------------------------------------------
Author: Luke Fraser (luke@abundantrobotics.com)
--------------------------------------------------------------------------------
'''
import math
import re
import numpy as np
from numpy import asmatrix as mat
from geometry_msgs.msg import Quaternion
from collections import namedtuple


AxisAngle = namedtuple('AxisAngle', 'x, y, z, angle')


class Quat(object):
    '''
    Class: Quat
    Description:
        - Quat is a simple quaternion class for performing simple quaternion
        operations. The quaternion follows ros convention and is ordered:
            X,Y,Z,W
        - The supported operations are:
            * Add - Q1 + Q2
            * Sub - Q1 - q2
            * Mul - Q1 * Q2
            * DIV - Q1 / Q2

        - Quat also has several convienance functions and properties:
            @Property
            * conjugate         - compute the conjugate of the quaternion
            * norm              - Compute the norm of the quaternion
            * inv               - Compute the inverse of the quaternion
            * x, y, z, w        - Element accessor and setters
            * msg               - Quaternion as ros msg
            @Method
            * copy              - create a copy of the current quaternion
            @StaticMethod
            * from_to_vectors   - compute quaternion from two vectors

        - Quat can also be accessed as an array
    '''
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.values = np.array([x, y, z, w])

    def __add__(self, q):
        '''
        quaternion addition operator. Supported addition operations:
            - Quat + Quat
            - Quat + int
            - Quat + float
        '''
        if isinstance(q, Quat):
            return Quat(*(self.values + q.values))
        if isinstance(q, float) or isinstance(q, int):
            return Quat(*(self.values + q))
        raise TypeError('Quat only adds with quat')

    def __sub__(self, q):
        '''
        quaternion subtraction operator. Supported subtraction operations:
            - Quat - Quat
            - Quat - int
            - Quat - float
        '''
        if isinstance(q, Quat):
            return Quat(*(self.values - q.values))
        if isinstance(q, float) or isinstance(q, int):
            return Quat(*(self.values - q))
        raise TypeError('Quat only subtracts with quat')

    def __mul__(self, q):
        '''
        quaternion multiply operator. Supported multiplication operations:
            - Quat * Quat
            - Quat * int
            - Quat * float
            - Quat * ndarray(vector3)
        '''
        if isinstance(q, Quat):
            x = self
            result = Quat(
                x[3]*q[0] + x[0]*q[3] + x[1]*q[2] - x[2]*q[1],
                x[3]*q[1] - x[0]*q[2] + x[1]*q[3] + x[2]*q[0],
                x[3]*q[2] + x[0]*q[1] - x[1]*q[0] + x[2]*q[3],
                x[3]*q[3] - x[0]*q[0] - x[1]*q[1] - x[2]*q[2]
            )
            return result
        if isinstance(q, int) or isinstance(q, float):
            return Quat(*(self.values * q))
        if isinstance(q, np.ndarray):
            return self.__mul__(Quat(q[0], q[1], q[2], 0.0))
        raise TypeError('quat not defined for type: %s' % type(q))

    def __truediv__(self, q):
        '''
        quaternion division. supported division operations
            - Quat / Quat       = Quat * Quat.inv
            - Quat / float      = Quat * (1.0 / float)
            - Quat / int        = Quat * (1.0 / float(int))
        '''
        if isinstance(q, Quat):
            return self.__mul__(q.inv)
        if isinstance(q, float) or isinstance(q, int):
            return self.__mul__(1.0 / q)
        raise TypeError('division not deined for type: %s' % type(q))
    
    def __div__(self, q):
        return self.__truediv__(q)

    @property
    def conjugate(self):
        '''
        compute the conjugate of the quaternion
        '''
        return Quat(-self[0], -self[1], -self[2], self[3])

    @property
    def norm(self):
        '''
        compute the norm of the quaternion
        '''
        return math.sqrt(np.sum(self.values * self.values))

    @property
    def inv(self):
        '''
        compute the inverse of the quanternion
        '''
        return self.conjugate * (1 / math.pow(self.norm, 2.0))

    def dot(self, quat):
        '''
        compute the dot product of the quaternion with another quaternion
        '''
        return np.dot(self.values, quat.values)

    @property
    def imag(self):
        '''
        return the imaginary component of the quaternion
        '''
        return self.values[:-1]

    @property
    def real(self):
        '''
        return the real component of the quaternion
        '''
        return self.values[-1]

    @property
    def x(self):
        '''
        x component property getter/setter
        '''
        return self[0]

    @x.setter
    def x(self, value):
        '''
        x component property getter/setter
        '''
        self.values[0] = value

    @property
    def y(self):
        '''
        y component property getter/setter
        '''
        return self[1]

    @y.setter
    def y(self, value):
        '''
        y component property getter/setter
        '''
        self.values[1] = value

    @property
    def z(self):
        '''
        z component property getter/setter
        '''
        return self[2]

    @z.setter
    def z(self, value):
        '''
        z component property getter/setter
        '''
        self.values[2] = value

    @property
    def w(self):
        '''
        w component property getter/setter
        '''
        return self[3]

    @w.setter
    def w(self, value):
        '''
        w component property getter/setter
        '''
        self.values[3] = value

    @property
    def i(self):
        '''
        i component property getter/setter
        '''
        return self[0]

    @i.setter
    def i(self, value):
        '''
        i component property getter/setter
        '''
        self.values[0] = value

    @property
    def j(self):
        '''
        j component property getter/setter
        '''
        return self[1]

    @j.setter
    def j(self, value):
        '''
        j component property getter/setter
        '''
        self.values[1] = value

    @property
    def k(self):
        '''
        k component property getter/setter
        '''
        return self[2]

    @k.setter
    def k(self, value):
        '''
        k component property getter/setter
        '''
        self.values[2] = value

    @property
    def msg(self):
        '''
        convert the quaternion into a supported ros quaternion message.
        '''
        q = Quaternion()
        q.x = self.x
        q.y = self.y
        q.z = self.z
        q.w = self.w
        return q

    def copy(self):
        '''
        copy quaternion.
        '''
        return Quat(self.x, self.y, self.z, self.w)

    @staticmethod
    def from_two_vectors(u, v):
        '''
        Generate a rotational quaternion between two vectors.
        '''
        vector_dot = np.dot(u, v)
        if vector_dot >= 1:
            identity_rotation = Quat(0, 0, 0, 1)
            return identity_rotation
        elif vector_dot <= -1:
            orthogonal_axis = np.cross(u, np.array([1, 0, 0]))
            if np.linalg.norm(orthogonal_axis) == 0:
                orthogonal_axis = np.cross(u, np.array([0, 1, 0]))
            semi_circle_rotation = Quat(orthogonal_axis[0], orthogonal_axis[1], orthogonal_axis[2], 0)
            semi_circle_rotation = semi_circle_rotation / semi_circle_rotation.norm
            return semi_circle_rotation
        else:
            w = np.cross(u, v)
            q = Quat(w[0], w[1], w[2], 1.0 + vector_dot)
            return q * (1 / q.norm)

    @property
    def skew_sym(self):
        '''
        return the skew symetric matrix of the quaternion.
        This provides the cross product as a matrix.
        '''
        return np.array([
            [0      , -self.k, self.j ],
            [self.k , 0      , -self.i],
            [-self.j, self.i , 0      ]
        ])

    def to_euler(self):
        ysqr = self.y * self.y
        t_0 = 2.0 * (self.w * self.x + self.y * self.z)
        t_1 = 1.0 - 2.0 * (self.x * self.x + ysqr)
        roll = math.atan2(t_0, t_1)

        t2 = 2.0 * (self.w * self.y - self.z * self.x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = 2.0 * (self.w * self.z + self.x * self.y)
        t4 = 1.0 - 2.0 * (ysqr + self.z * self.z)
        yaw = math.atan2(t3, t4)
        return np.array([roll, pitch, yaw])


    @staticmethod
    def from_euler(roll, pitch, yaw):
        t_0 = math.cos(yaw * 0.5)
        t_1 = math.sin(yaw * 0.5)
        t_2 = math.cos(roll * 0.5)
        t_3 = math.sin(roll * 0.5)
        t_4 = math.cos(pitch * 0.5)
        t_5 = math.sin(pitch * 0.5)

        w = t_0 * t_2 * t_4 + t_1 * t_3 * t_5
        x = t_0 * t_3 * t_4 - t_1 * t_2 * t_5
        y = t_0 * t_2 * t_5 + t_1 * t_3 * t_4
        z = t_1 * t_2 * t_4 - t_0 * t_3 * t_5
        return Quat(x, y, z, w)

    def to_matrix(self):
        r = (
            (2*self.w * self.w - 1) * np.identity(3) +
            2 * self.w * self.skew_sym +
            2 * mat(self.imag).T * mat(self.imag)
        )
        return r

    @staticmethod
    def from_matrix(R):
        w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
        w4 = w * 4.0
        x = (R[2,1] - R[1,2]) / w4
        y = (R[0,2] - R[2,0]) / w4
        z = (R[1,0] - R[0,1]) / w4
        return Quat(x, y, z, w)

    def to_axis_angle(self):
        angle = 2.0 * math.acos(self.w)
        w_div = np.sqrt(1.0 - self.w * self.w)
        if w_div < 0.001:
            x = 1.0
            y = 0.0
            z = 0.0
        else:
            x = self.x / w_div
            y = self.y / w_div
            z = self.z / w_div
        return AxisAngle(x, y, z, angle)

    @staticmethod
    def from_axis_angle(A):
        sin_a = np.sin(A.angle / 2.0)
        x = A.x * sin_a
        y = A.y * sin_a
        z = A.z * sin_a
        w = np.cos(A.angle / 2.0)
        return Quat(x, y, z, w)


    @staticmethod
    def from_string(quat_str):
        expr = (
            '^\['
            'x: *(-*[0-9]+[\.]*[0-9]*), *'
            'y: *(-*[0-9]+[\.]*[0-9]*), *'
            'z: *(-*[0-9]+[\.]*[0-9]*), *'
            'w: *(-*[0-9]+[\.]*[0-9]*)'
            '\]$')
        match = re.search(expr, quat_str)
        try:
            result = Quat(
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)))
        except AttributeError:
            raise AttributeError(
                'Could not determin quaternion from: %s' % (quat_str))
        if not result:
            raise AttributeError(
                'Could not read quaternion from string: %s' % (quat_str))
        return result

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self):
        return '[x: %f, y: %f, z: %f, w: %f]' % (
            self.x, self.y, self.z, self.w)




class SimilarQuaternionAverager(object):
    '''
    class SimilarQuaternionAverager
    Description: Averages N similar quaternions together. This is useful to get
        an estimate of fused quaternions.
    '''
    def __init__(self, initial_quaternion=None):
        self.initial_quaternion = initial_quaternion
        self.cummulative = None if not initial_quaternion else\
            initial_quaternion.copy()
        self.seq_num = 0 if not self.initial_quaternion else 1
        self.averaged_seq = None
        self._result = None

    def add_sample(self, sample_quat):
        '''
        Add a new sample to the estimate model
        '''
        if self.seq_num == 0:
            self.initial_quaternion = sample_quat.copy()
            self.cummulative = sample_quat.copy()
            self.seq_num += 1
            return
        # Average the input into cummulative
        if not self._is_quat_close_to_initial(sample_quat):
            sample_quat = sample_quat * -1.0
        self.cummulative += sample_quat
        self.seq_num += 1

    def _is_quat_close_to_initial(self, quat):
        '''
        check if initial quaternion atitude is close to another quaternion
        '''
        dot = self.initial_quaternion.dot(quat)
        if dot < 0.0:
            return False
        return True

    @property
    def average(self):
        '''
        returns the estimate of the average quaternion.
        '''
        if self.cummulative and self.averaged_seq != self.seq_num:
            result = (self.cummulative * (1.0 / self.seq_num))
            result = result / result.norm
            self.averaged_seq = self.seq_num
            self._result = result
            return result
        elif self.averaged_seq == self.seq_num:
            return self._result
        else:
            return Quat(0.0, 0.0, 0.0, 1.0)
