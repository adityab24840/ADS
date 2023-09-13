#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Node of a doubly linked list.
struct Node {
    int key;
    struct Node * next; // Pointer to next node in DLL.
    struct Node * prev; // Pointer to previous node in DLL.
};

struct LinkedList {
    struct Node * head;
    struct Node * tail;
};

typedef struct Node Node;
typedef struct LinkedList LinkedList;

// Function Prototype or method on linked list.
LinkedList * create_list(void); // Initiate a linked list.
unsigned is_empty(LinkedList *); // Check if a linked list is empty or no.
void insert_first(LinkedList *, int); // Add a node in the beginning.
void insert_last(LinkedList *, int); // Add a node in the end.
void insert_after(LinkedList *, Node *, int key); // Add a node after a given node.
void insert_before(LinkedList *, Node *, int key); // Add a node before a given node.
Node * find_node(Node *, int); // Check if a given key exist or no.
void show(Node *); // Higher order function to be used in traverse: show node's key.
void traverse_forward(Node *, void (*callback)(Node *)); // Forward Traverse the linked list starting from a given Node.
void traverse_backward(Node *, void (*callback)(Node *)); // Backward Traverse the linked list starting from a given Node.
void remove_succesor(Node *); // Remove the successor of a Node.
void remove_node(Node *); // Remove Node from the list.
void remove_head(LinkedList *); // Remove a node from the beginning.
void remove_tail(LinkedList *); // Remove a node from the end.

LinkedList * create_list() {
    LinkedList *ll ;
    ll = malloc(sizeof(LinkedList)) ;
    ll -> head = NULL ;
    ll -> tail = NULL ;
    return ll ;
}

unsigned is_empty(LinkedList * ll) {
    return (ll -> head == NULL) && (ll -> tail == NULL);
}

void insert_first(LinkedList * ll, int key) {
    // Dynamic allocate node.
    Node * nd;
    nd = malloc(sizeof(Node));
    // put in the data.
    nd -> key = key;
    // Make next of new node as head.
    nd -> next = ll -> head;
    // Make previous of new node as NULL.
    nd -> prev = NULL;
    // If list is empty make the tail point to the new node.
    if (is_empty(ll)) 
        ll -> tail = nd;
    else
        // change prev of head node to new node.
        ll -> head -> prev = nd;
    // move the head to point to the new node.
    ll -> head = nd;
}

void insert_last(LinkedList * ll, int key) {
    // Dynamic allocate node.
    Node * nd;
    nd = malloc(sizeof(Node));
    // put in the data.
    nd -> key = key;
    // This new node is going to be the last node, so next of it is NULL.
    nd -> next = NULL;
    // If the Linked List is empty, then the new node is head and tail.
    if (is_empty(ll)) {
        nd -> prev = NULL;
        ll -> head = nd; 
        ll -> tail = nd;
    } else {
        // Make previous of new node as tail.
        nd -> prev = ll -> tail;
        // change next of tail node to new node.
        ll -> tail -> next = nd;
        // move the prev to point to the new node.
        ll -> tail = nd; 
    }
    

}

void insert_after(LinkedList * ll, Node * nd, int key) {
    // Dynamic allocate node.
    Node * temp_nd;
    temp_nd = malloc(sizeof(Node));
    // put in the data.
    temp_nd -> key = key;
    // Make next of new node as next of nd.
    temp_nd -> next = nd -> next;
    // Change previous of new nd's next node to the new node.
    if (temp_nd -> next != NULL)
        nd -> next -> prev = temp_nd;
    
    // Make the next of nd as new node.
    nd -> next = temp_nd;
    // Make nd as previous of new node.
    temp_nd -> prev = nd;
    if (temp_nd -> next == NULL)
        // If the next of new node is NULL, it will be  the new tail node.
        ll -> tail = temp_nd;
}

void insert_before(LinkedList * ll, Node * nd, int key) {
    // Dynamic allocate node.
    Node * temp_nd;
    temp_nd = malloc(sizeof(Node));
    // put in the data.
    temp_nd -> key = key;
    // Make prev of new node as prev of nd.
    temp_nd -> prev = nd -> prev;
    // Change next of nd's previous node to the new node.
    if (temp_nd -> prev != NULL)
        nd -> prev -> next = temp_nd;
    // Make the prev of nd as new node.
    nd -> prev = temp_nd;
    // Make nd as next of new node.
    temp_nd -> next = nd;

    if (temp_nd -> prev == NULL)
        // If the previous of new node is NULL, it will be  the new head node.
        ll -> head = temp_nd;
}

Node * find_node(Node * nd, int key) {
    while (nd && nd -> key != key) {
        nd = nd -> next;
    }
    return nd;
}

void traverse_forward(Node * nd, void(*callback)(Node *)) {
    while (nd) {
        (*callback)(nd);
        nd = nd -> next;
    }
}

void traverse_backward(Node * nd, void(*callback)(Node *)) {
    while (nd) {
        (*callback)(nd);
        nd = nd -> prev;
    }
}

void show(Node * nd) {
    printf("%d -- ", nd -> key);
}

void remove_succesor(Node * nd) {
    Node * temp_nd;
    temp_nd = nd -> next;
    nd -> next = temp_nd -> next;
    temp_nd -> next -> prev = nd;
    free(temp_nd);
}

void remove_node(Node * nd) {
    if (nd -> next != NULL)
        nd -> next -> prev = nd -> prev;
    if (nd -> prev != NULL)
        nd -> prev -> next = nd -> next;
    free(nd);
}

void remove_head(LinkedList * ll) {
    Node * nd;
    nd = ll -> head;
    ll -> head = nd -> next;
    free(nd);
    if (ll -> head == NULL) 
        ll -> tail = NULL;
    else 
        ll -> head -> prev = NULL;
}

void remove_tail(LinkedList * ll) {
    Node * nd;
    if (ll -> head == ll -> tail) 
        remove_head(ll);
    else {
        nd = ll -> head; 
        while(nd ->next != ll -> tail)
            nd = nd -> next;
        nd -> next = NULL;
        free(ll -> tail);
        ll -> tail = nd;
    }
}