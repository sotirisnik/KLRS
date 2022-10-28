import numpy as np
from sortedcontainers import SortedList

def random_number( a, b, rng ):
    return int( (b-a) * rng.random() + a )

def random_real( a, b, rng ):
    return float( (b-a) * rng.random() + a )

def print_seq( x ):
    for i in x:
        print( i, end=' ' )
    print()
    
def KL( P, Q ):
    ret = 0.0
    for i in range( len(P) ):
        if P[i] > 0.0:
            ret += P[i] * ( np.log( P[i] ) - np.log( Q[i] ) )
    return ret

def softmax( x ):
    m = np.max( x )
    p = np.exp( x - m )
    return p / np.sum(p)

class Buffer:

    def __init__(self, capacity=5000, seed=1 ):
        self.q = []
        self.total_items = 0
        self.max_cap = capacity
        self.visit = 0
        self.rng = np.random.default_rng(seed)
        self.m = {}#buffer counts
        self.n = {}#stream counts

    def Insert():
        pass
    
    def UniformSample( self, size=128 ):
        perm = self.rng.permutation( self.total_items )[:size]
        return [ self.q[i] for i in perm ]
    
    def WeightedSample( self, size=128 ):
        if self.total_items == 0:
            return []
        b = -1.0
        x = []
        for data in self.q:
            x.append( self.m[ data[1] ] )
        x = np.array(x,dtype=np.float64)
        p = (x)**b / ((x)**b).sum()
        subset = self.rng.choice( self.total_items, min(self.total_items,size), replace=False, p=p )
        return [ self.q[i] for i in subset ]

def Increase( counts, label ):
    if label not in counts:
        counts[ label ] = 1
    else:
        counts[ label ] += 1

def Decrease( counts, label ):
    if label not in counts:
        counts[ label ] = 0
    else:
        counts[ label ] = max( 0, counts[label]-1)

def AddPosToCategory( dictionary, label, pos ):
    if label not in dictionary:
        dictionary[label] = {pos}
    else:
        dictionary[label].add( pos )

def RemovePosFromCategory( dictionary, label, pos ):
    dictionary[label].remove( pos )    

class ReservoirSampling(Buffer):
    
    def __init__(self, capacity=5000, seed=1 ):
        Buffer.__init__(self, capacity, seed )
        
    def Insert( self, data ):
        self.visit += 1

        #number of instances encounters thus far per class
        Increase( self.n, data[1] )

        if self.total_items+1 <= self.max_cap:
            self.total_items += 1
            self.q.append( data )
            Increase( self.m, data[1] )
            return self.total_items-1
        else:
            j = random_number( 0, self.visit, self.rng )#[0,self.visit)
            if j <= self.max_cap-1:
                Decrease( self.m, self.q[j][1] )#decrease counter of removing item
                self.q[j] = data
                Increase( self.m, data[1] )
                return j
            return -1

class Random(Buffer):
    
    def __init__(self, capacity=5000, seed=1 ):
        Buffer.__init__(self, capacity, seed )
        
    def Insert( self, data ):
        self.visit += 1

        #number of instances encounters thus far per class
        Increase( self.n, data[1] )

        if self.total_items+1 <= self.max_cap:
            self.total_items += 1
            self.q.append( data )
            Increase( self.m, data[1] )
            return self.total_items-1
        else:
            j = random_number( 0, 1, self.rng )
            if j == 1:
                pos = random_number( 0, self.total_items-1, self.rng )
                Decrease( self.m, self.q[pos][1] )#decrease counter of removing item
                self.q[pos] = data
                Increase( self.m, data[1] )
                return j
            return -1

class WRS(Buffer):
    
    def __init__(self, capacity=5000, seed=1 ):
        Buffer.__init__(self, capacity, seed )
        self.wSum = 0.0
        
    def Insert( self, data ):
        self.visit += 1

        #number of instances encounters thus far per class
        Increase( self.n, data[1] )

        if self.total_items+1 <= self.max_cap:
            self.total_items += 1
            self.q.append( data )
            Increase( self.m, data[1] )
            self.wSum += data[2]#contains the weight of the data point
            return self.total_items-1
        else:
            j = random_real( 0, 1, self.rng )#[0,self.visit)
            extend_wSum = self.wSum + data[2]
            p = min( data[2] / extend_wSum, 1.0/self.total_items  )
            if j <= p:
                #print( j, len(self.q), self.max_cap, self.q )
                pos = random_number( 0, self.total_items-1, self.rng )
                Decrease( self.m, self.q[pos][1] )#decrease counter of removing item
                self.wSum -= self.q[pos][2]
                self.q[pos] = data
                Increase( self.m, data[1] )
                self.wSum += data[2]
                return pos
            return -1

class CBRS(Buffer):#custom cbrs with optimized insertions
    
    def __init__(self, capacity=5000, seed=1 ):
        Buffer.__init__(self, capacity, seed )
        self.full_class = {}#contains classes that became full
        self.position_per_cat = {}#i.e. position_per_cat[2] contains a set of all indices for category 2
        self.group_by_freq = {}#group by freq, i.e. group_by_freq[264] contains all categories that have size 264
        self.multiset = SortedList([])

    def Insert( self, data ):
        self.visit += 1

        #number of instances encounters thus far per class
        Increase( self.n, data[1] )

        if self.total_items+1 <= self.max_cap:
            self.q.append( data )
            #stores frequency in mi for class==data[1], multiset is for knowing the maximum frequency at any time
            if data[1] not in self.m:
                self.m[ data[1] ] = 1
                self.multiset.add( 1 )
                if 1 not in self.group_by_freq:
                    self.group_by_freq[1] = { data[1] }
                else:
                    self.group_by_freq[1].add( data[1] )
                self.position_per_cat[ data[1] ] = { self.total_items }
            else:
                self.multiset.remove( self.m[ data[1] ] )
                self.group_by_freq[ self.m[ data[1] ] ].remove( data[1] )
                self.m[ data[1] ] += 1
                if self.m[ data[1] ] not in self.group_by_freq:
                    self.group_by_freq[ self.m[ data[1] ] ] = { data[1] }
                else:
                    self.group_by_freq[ self.m[ data[1] ] ].add( data[1] )
                self.multiset.add( self.m[ data[1] ] )
                self.position_per_cat[ data[1] ].add( self.total_items )

            #input()
            self.total_items += 1
            
            #when buffer is mark full class
            if self.total_items == self.max_cap:
                maxu = self.multiset[-1]
                for mKey in self.m:
                    if self.m[mKey] == maxu:
                        self.full_class[mKey] = True

            return self.total_items-1
        else:

            if data[1] not in self.full_class:
                largest_class_num = self.multiset[-1]
                #all we need is to pick an instance and overwrite, so first we randomly pick one of the largest class at random
                cj = self.rng.choice( tuple( self.group_by_freq[largest_class_num] ), 1 ).item()
                #now pick a position of an instance from this class and remove it
                buffer_position = self.rng.choice( tuple( self.position_per_cat[ cj ]), 1 ).item()
                
                #start remove element
                self.multiset.remove( self.m[ self.q[buffer_position][1] ] )
                self.group_by_freq[ self.m[ self.q[buffer_position][1] ] ].remove( self.q[buffer_position][1] )
                self.m[ self.q[buffer_position][1] ] = max( 0, self.m[ self.q[buffer_position][1] ]-1 )#decrease counter of the class for the instance we are about to ovewrite
                self.position_per_cat[ self.q[buffer_position][1] ].remove( buffer_position )
                
                if self.m[ self.q[buffer_position][1] ] not in self.group_by_freq:
                    self.group_by_freq[ self.m[ self.q[buffer_position][1] ] ] = { self.q[buffer_position][1] }
                else:
                    self.group_by_freq[ self.m[ self.q[buffer_position][1] ] ].add( self.q[buffer_position][1] )
                self.multiset.add( self.m[ self.q[buffer_position][1] ] )
                #end remove element

                #then insert the new item
                if data[1] not in self.m:
                    self.m[ data[1] ] = 1
                    self.multiset.add( 1 )
                    if 1 not in self.group_by_freq:
                        self.group_by_freq[1] = { data[1] }
                    else:
                        self.group_by_freq[1].add( data[1] )
                    self.position_per_cat[ data[1] ] = { buffer_position }
                else:
                    self.multiset.remove( self.m[ data[1] ] )
                    self.group_by_freq[ self.m[ data[1] ] ].remove( data[1] )
                    self.m[ data[1] ] += 1
                    if self.m[ data[1] ] not in self.group_by_freq:
                        self.group_by_freq[ self.m[ data[1] ] ] = { data[1] }
                    else:
                        self.group_by_freq[ self.m[ data[1] ] ].add( data[1] )
                    self.multiset.add( self.m[ data[1] ] )
                    self.position_per_cat[ data[1] ].add( buffer_position )                
                
                self.q[buffer_position] = data

                #when buffer is mark full class
                if self.total_items == self.max_cap:
                    maxu = self.multiset[-1]
                    for mKey in self.m:
                        if self.m[mKey] == maxu:
                            self.full_class[mKey] = True

                return buffer_position

            else:
                u = random_real( 0.0, 1.0, self.rng )#random number in [0,1]
                frac = self.m[ data[1] ]  / self.n[ data[1] ]
                if u <= frac:
                    buffer_position = self.rng.choice( tuple( self.position_per_cat[data[1]] ), 1 ).item()
                    self.q[buffer_position] = data
                    return buffer_position
            
            return -1

class KLRS(Buffer):

    def __init__(self, capacity=5000, seed=1, num_classes=10, pi=0.5, kl_alpha=0.5, kl_technique='default' ):
        Buffer.__init__(self, capacity, seed )
        self.num_classes = num_classes
        self.pi = pi
        self.position_per_cat = {}#i.e. position_per_cat[2] contains a set of all indices for category 2
        self.uniform = [1.0/num_classes]*num_classes#uniform
        self.kl_alpha = kl_alpha
        self.kl_technique = kl_technique

    def Insert( self, data ):
        
        Increase( self.n, data[1] )

        if self.total_items+1 <= self.max_cap:
            self.q.append( data )
            self.total_items += 1
            Increase( self.m, data[1] )
            AddPosToCategory( self.position_per_cat, data[1], self.total_items-1 )
            return

        if self.kl_technique == 'default':
            stream_prob = self.GetStreamProb()
        else:
            target_prob = self.GetTargetProb()

        P_cnt = self.GetDistribution()
        actions = []
        #BEGIN OF ALL POSSIBLE ACTIONS, check each class that has at least 1 counter
        for cj in range(self.num_classes):
            if P_cnt[cj] >= 1:
                memory_counts = np.array( P_cnt, dtype=np.float64 ).copy()
                memory_counts[ cj ] -= 1
                memory_counts[ data[1] ] += 1
                memory_prob = (memory_counts)/np.sum(memory_counts)
                
                inf_cost_detected = False
                if memory_counts[ cj ] == 0:
                    inf_cost_detected = True
                
                if inf_cost_detected == False:
                    if self.kl_technique == 'default':
                        actions.append( { 'label': cj, 'cost': (1-self.pi) * KL(self.uniform,memory_prob) + self.pi * KL(stream_prob,memory_prob) } )
                    else:
                        actions.append( { 'label': cj, 'cost': KL(target_prob,memory_prob) } )
        #END OF ALL POSSIBLE ACTIONS
        
        minu_idx = 0
        for j in range( len(actions) ):
            if actions[minu_idx]['cost'] > actions[j]['cost']:
                minu_idx = j
            elif actions[minu_idx]['cost'] == actions[j]['cost'] and actions[j]['label'] == data[1]:
                minu_idx = j
                
        if actions[minu_idx]['label'] == data[1]:
            #class-specific reservoir sampling
            u = random_number( 1, self.n[ data[1] ]+1, self.rng )
            if u <= self.max_cap:
                c =  tuple( self.position_per_cat[ data[1] ] )
                pos = self.rng.choice( c )
                self.q[ pos ] = data
        else:
            min_classes = set( [ actions[j]['label'] for j in range( len(actions) ) if actions[minu_idx]['cost'] == actions[j]['cost'] ] )
            random_min_pick = self.rng.choice( tuple(min_classes) )#choose a class at random from min_classes
            c = tuple( self.position_per_cat[ random_min_pick ] )#get all indices for the random_picked_class
            pos = self.rng.choice( c )
            Decrease( self.m, self.q[pos][1] )#decrease counter of removing item
            RemovePosFromCategory( self.position_per_cat, self.q[pos][1], pos )
            removed_class = self.q[pos][1]
            self.q[ pos ] = data
            AddPosToCategory( self.position_per_cat, self.q[pos][1], pos )
            Increase( self.m, data[1] )
            #P_cnt = self.GetDistribution()
            #print( P_cnt, 'insert class', data[1] )
            #print( 'removed', removed_class)
            
    def GetStreamProb(self):
        stream_prob = np.full(self.num_classes,0.0)
        for i in self.n:
            stream_prob[i] += self.n[i]
        stream_prob = stream_prob / stream_prob.sum()
        return stream_prob

    def GetTargetProb(self):
        target_prob = self.GetStreamProb()
        for i in range( self.num_classes ):
            if target_prob[i] > 0.0:
                target_prob[i] = self.kl_alpha * np.log(target_prob[i])
            else:
                target_prob[i] = -np.inf
        target_prob = softmax( target_prob )#e^{ log(pi)^alpha } or e^{-inf}=0.0
        return target_prob

    def CalculatePi(self, test_prob):
        #print( 'stream prob', stream_prob )
        #rint( 'test prob', test_prob )
        #print( 'uniform prob', self.uniform )
        stream_prob = self.GetStreamProb()
        print( 'stream', stream_prob )
        print( 'test', test_prob )
        d1, d2 = KL(test_prob,stream_prob), KL(test_prob,self.uniform) 
        pi = d1 / (d1+d2)
        return pi

    def SetPi( self, pi ):
        self.pi = pi

    def GetDistribution(self):
        counters = np.full( self.num_classes, 0 )#[0]*self.num_classes
        for i in self.m:
            counters[ i ] += self.m[i]
        return counters